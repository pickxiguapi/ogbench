import unittest
from collections import deque

import jax
import jax.numpy as jnp
import numpy as np

from latent_subgoal_runtime import LatentSubgoalGenerator
from lewm_jax.planner import (
    JAXLeWMCEMPolicy,
    StagedLeWMCEMPolicy,
    reduce_rollout_costs,
    subgoal_planning_horizon,
)


class FakeChunkAgent:
    action_horizon = 5

    def sample_actions(self, observations, goals, seed, temperature):
        assert observations.shape[0] == 1
        assert goals.shape[0] == 1
        assert temperature == 0.0
        return jnp.arange(10, dtype=jnp.float32)[None]


class FakeLatentGoalChunkAgent(FakeChunkAgent):
    def __init__(self):
        self.latent_goals = None

    def sample_actions(self, observations, goals, seed, temperature):
        raise AssertionError('Final-goal policy path must not be used with a subgoal.')

    def sample_actions_with_latent_goal(
        self, observations, latent_goals, seed, temperature
    ):
        assert observations.shape[0] == 1
        assert latent_goals.shape == (1, 3)
        assert temperature == 0.0
        self.latent_goals = np.asarray(latent_goals)
        return jnp.arange(10, dtype=jnp.float32)[None]


class FakeLatentPopulationAgent(FakeChunkAgent):
    def sample_actions_with_latent_goal(
        self, observations, latent_goals, seed, temperature
    ):
        assert observations.shape[0] == latent_goals.shape[0]
        return jnp.full(
            (observations.shape[0], 10), temperature, dtype=jnp.float32
        )


class FakeScaler:
    action_dim = 2
    mean = np.array([1.0, -2.0], dtype=np.float32)
    scale = np.array([2.0, 4.0], dtype=np.float32)

    def transform(self, value):
        return (np.asarray(value) - self.mean) / self.scale


class FakePlanner:
    def __init__(self, value, *, use_subgoal):
        self.value = value
        self.subgoal_generator = object() if use_subgoal else None
        self.action_block = 5
        self.horizon = 2 if use_subgoal else 5
        self.lewm_checkpoint = '/tmp/shared.msgpack'
        self.calls = 0

    def reset(self, action_space, num_envs):
        self.num_envs = num_envs

    def get_actions(self, pixels, goals, alive):
        self.calls += 1
        return np.full((len(alive), 2), self.value, dtype=np.float32)


class FakeWorldModel:
    def _rollout_predictions(self):
        raise AssertionError('Fake apply should receive, but not call, this method.')

    def apply(self, variables, pixels, goals, candidates, method=None):
        del variables, pixels, goals, method
        displacement = jnp.mean(candidates, axis=-1, keepdims=True)
        predictions = jnp.repeat(
            jnp.cumsum(displacement, axis=-2), 3, axis=-1
        ).astype(jnp.bfloat16)
        return jnp.zeros((1, 3), dtype=jnp.float32), predictions


def guidance_policy():
    policy = object.__new__(JAXLeWMCEMPolicy)
    policy.horizon = 5
    policy.action_block = 5
    policy.atomic_action_dim = 2
    policy.block_action_dim = 10
    policy.guidance_policy = FakeChunkAgent()
    policy.subgoal_generator = None
    policy.guidance_goal_mode = 'subgoal'
    policy.guidance_action_space = 'planner'
    policy.scaler = FakeScaler()
    policy.warm_starts = [np.full((2, 10), -3.0, dtype=np.float32)]
    return policy


class PlannerTest(unittest.TestCase):
    def test_candidate_trace_returns_the_exact_last_cem_pool(self):
        policy = object.__new__(JAXLeWMCEMPolicy)
        policy.model = FakeWorldModel()
        policy.variables = {}
        policy.num_samples = 6
        policy.iterations = 2
        policy.topk = 2
        policy.var_scale = 1.0
        policy.cost_mode = 'last'
        policy.subgoal_generator = None
        policy.guidance_mode = 'none'
        policy.guidance_population_size = 0
        policy.guidance_first_block_std = None
        policy.planner_action_low = None
        policy.planner_action_high = None
        policy.trace_candidates = True
        policy.lewm_config = {'embed_dim': 3}
        policy.horizon = 2
        plan_one = jax.jit(policy._build_plan_one())

        output = plan_one(
            jax.random.PRNGKey(0),
            jnp.zeros((1, 4, 4, 3), dtype=jnp.uint8),
            jnp.zeros((1, 4, 4, 3), dtype=jnp.uint8),
            jnp.zeros((3,), dtype=jnp.float32),
            jnp.zeros((2, 4), dtype=jnp.float32),
            jnp.zeros((1, 4), dtype=jnp.float32),
        )

        self.assertEqual(len(output), 5)
        self.assertEqual(output[0].shape, (2, 4))
        self.assertEqual(output[2].shape, (6, 2, 4))
        self.assertEqual(output[3].shape, (6, 2, 3))
        self.assertEqual(output[3].dtype, jnp.float32)
        self.assertEqual(output[4].shape, (6,))

        policy.trace_candidates = False
        regular_output = jax.jit(policy._build_plan_one())(
            jax.random.PRNGKey(0),
            jnp.zeros((1, 4, 4, 3), dtype=jnp.uint8),
            jnp.zeros((1, 4, 4, 3), dtype=jnp.uint8),
            jnp.zeros((3,), dtype=jnp.float32),
            jnp.zeros((2, 4), dtype=jnp.float32),
            jnp.zeros((1, 4), dtype=jnp.float32),
        )
        self.assertEqual(len(regular_output), 2)

    def test_subgoal_planner_can_guide_policy_with_final_goal(self):
        policy = guidance_policy()
        policy.subgoal_generator = object()
        policy.guidance_goal_mode = 'final'
        pixels = np.zeros((1, 16, 16, 3), dtype=np.uint8)

        block = policy._guidance_block(
            pixels,
            pixels,
            jax.random.PRNGKey(0),
            target_embedding=np.array([1.0, 2.0, 3.0], dtype=np.float32),
        )

        np.testing.assert_array_equal(block, np.arange(10, dtype=np.float32))

    def test_actor_population_contains_mode_and_stochastic_blocks(self):
        policy = guidance_policy()
        policy.subgoal_generator = object()
        policy.guidance_policy = FakeLatentPopulationAgent()
        policy.guidance_population_size = 4
        policy.guidance_temperature = 0.3
        pixels = np.zeros((1, 16, 16, 3), dtype=np.uint8)
        latent_target = np.array([1.0, 2.0, 3.0], dtype=np.float32)

        blocks = policy._guidance_population(
            pixels,
            pixels,
            jax.random.PRNGKey(0),
            target_embedding=latent_target,
        )

        np.testing.assert_array_equal(blocks[0], np.zeros(10, dtype=np.float32))
        np.testing.assert_allclose(blocks[1:], 0.3)

    def test_staged_planner_switches_on_an_action_block_boundary(self):
        local = FakePlanner(1.0, use_subgoal=True)
        final = FakePlanner(2.0, use_subgoal=False)
        policy = StagedLeWMCEMPolicy(local, final, switch_after_steps=10)
        policy.reset(type('Space', (), {'shape': (2,)})(), 2)
        pixels = np.zeros((2, 1, 4, 4, 3), dtype=np.uint8)
        alive = np.ones(2, dtype=bool)

        for _ in range(10):
            np.testing.assert_array_equal(
                policy.get_actions(pixels, pixels, alive),
                np.ones((2, 2), dtype=np.float32),
            )
        np.testing.assert_array_equal(
            policy.get_actions(pixels, pixels, alive),
            np.full((2, 2), 2.0, dtype=np.float32),
        )
        self.assertEqual(local.calls, 10)
        self.assertEqual(final.calls, 1)

    def test_subgoal_horizon_and_cost_reduction(self):
        self.assertEqual(subgoal_planning_horizon(10, 5), 2)
        with self.assertRaisesRegex(ValueError, 'divisible'):
            subgoal_planning_horizon(7, 5)

        distances = jnp.asarray([[5.0, 1.0, 4.0], [2.0, 3.0, 6.0]])
        np.testing.assert_array_equal(
            reduce_rollout_costs(distances, 'last'), [4.0, 6.0]
        )
        np.testing.assert_array_equal(
            reduce_rollout_costs(distances, 'moh'), [1.0, 2.0]
        )
        np.testing.assert_allclose(
            reduce_rollout_costs(distances, 'path_mean'), [10.0 / 3.0, 11.0 / 3.0]
        )

    def test_subgoal_runtime_uses_three_encoded_history_frames(self):
        generator = object.__new__(LatentSubgoalGenerator)
        generator.history_size = 3
        generator.histories = [
            deque(
                [
                    np.full((2, 2, 1), 1.0, dtype=np.float32),
                    np.full((2, 2, 1), 2.0, dtype=np.float32),
                    np.full((2, 2, 1), 3.0, dtype=np.float32),
                ],
                maxlen=3,
            )
        ]
        generator.generation_counts = np.zeros(1, dtype=np.int64)
        generator.embed_dim = 3
        generator.seed = 42
        generator._requires_rng = False
        generator.encode_pixels = lambda pixels: np.repeat(
            np.asarray(pixels).mean(axis=(1, 2, 3))[:, None], 3, axis=1
        ).astype(np.float32)
        captured = {}

        def predict(history, goal):
            captured['history'] = np.asarray(history)
            return history[:, -1]

        generator._predict = predict
        target = generator.predict(0, np.full((2, 2, 1), 4.0, dtype=np.float32))

        self.assertEqual(captured['history'].shape, (1, 3, 3))
        np.testing.assert_array_equal(captured['history'][0, :, 0], [1.0, 2.0, 3.0])
        np.testing.assert_array_equal(target, [3.0, 3.0, 3.0])
        np.testing.assert_array_equal(generator.generation_counts, [1])

    def test_paired_plan_keys_match_with_and_without_guidance(self):
        vanilla = object.__new__(JAXLeWMCEMPolicy)
        guided = object.__new__(JAXLeWMCEMPolicy)
        for policy, agent in ((vanilla, None), (guided, FakeChunkAgent())):
            policy.seed = 42
            policy.paired_plan_keys = True
            policy.plan_counts = np.zeros(2, dtype=np.int64)
            policy.guidance_policy = agent

        vanilla_guidance_key, vanilla_plan_key = vanilla._next_plan_keys(1)
        guided_guidance_key, guided_plan_key = guided._next_plan_keys(1)

        np.testing.assert_array_equal(vanilla_plan_key, guided_plan_key)
        np.testing.assert_array_equal(vanilla_guidance_key, guided_guidance_key)

    def test_guidance_replaces_only_first_cem_block(self):
        policy = guidance_policy()
        pixels = np.zeros((1, 16, 16, 3), dtype=np.uint8)
        goals = np.zeros_like(pixels)

        initial = policy._initial_mean(
            0,
            pixels,
            goals,
            jax.random.PRNGKey(0),
        )

        np.testing.assert_array_equal(initial[0], np.arange(10, dtype=np.float32))
        np.testing.assert_array_equal(initial[1], np.full(10, -3.0, dtype=np.float32))
        np.testing.assert_array_equal(initial[2:], np.zeros((3, 10), dtype=np.float32))

    def test_subgoal_guidance_uses_the_cem_latent_target(self):
        policy = guidance_policy()
        policy.subgoal_generator = object()
        policy.guidance_policy = FakeLatentGoalChunkAgent()
        pixels = np.zeros((1, 16, 16, 3), dtype=np.uint8)
        final_goal_pixels = np.full_like(pixels, 255)
        latent_target = np.array([1.0, 2.0, 3.0], dtype=np.float32)

        initial = policy._initial_mean(
            0,
            pixels,
            final_goal_pixels,
            jax.random.PRNGKey(0),
            target_embedding=latent_target,
        )

        np.testing.assert_array_equal(
            policy.guidance_policy.latent_goals,
            latent_target[None],
        )
        np.testing.assert_array_equal(initial[0], np.arange(10, dtype=np.float32))

    def test_path_guidance_uses_the_terminal_latent_target(self):
        policy = guidance_policy()
        policy.subgoal_generator = object()
        policy.guidance_policy = FakeLatentGoalChunkAgent()
        pixels = np.zeros((1, 16, 16, 3), dtype=np.uint8)
        latent_path = np.array(
            [[-1.0, -2.0, -3.0], [1.0, 2.0, 3.0]], dtype=np.float32
        )

        policy._initial_mean(
            0,
            pixels,
            pixels,
            jax.random.PRNGKey(0),
            target_embedding=latent_path,
        )

        np.testing.assert_array_equal(
            policy.guidance_policy.latent_goals,
            latent_path[-1:]
        )

    def test_environment_action_guidance_is_standardized(self):
        policy = guidance_policy()
        policy.guidance_action_space = 'environment'
        pixels = np.zeros((1, 16, 16, 3), dtype=np.uint8)

        block = policy._guidance_block(
            pixels, pixels, jax.random.PRNGKey(0)
        )

        expected = FakeScaler().transform(
            np.arange(10, dtype=np.float32).reshape(5, 2)
        ).reshape(10)
        np.testing.assert_allclose(block, expected)

    def test_guidance_shape_is_checked(self):
        policy = guidance_policy()
        policy.block_action_dim = 9
        pixels = np.zeros((1, 16, 16, 3), dtype=np.uint8)

        with self.assertRaisesRegex(ValueError, 'Guidance policy returned'):
            policy._guidance_block(pixels, pixels, jax.random.PRNGKey(0))


if __name__ == '__main__':
    unittest.main()
