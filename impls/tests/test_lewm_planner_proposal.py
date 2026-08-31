import unittest

import jax
import jax.numpy as jnp
import numpy as np
from eval_lewm_4tasks import JAXLeWMCEMPolicy
from lewm_jax.planner import (
    fixed_subgoal_horizon_index,
    latent_path_waypoint_index,
    select_latent_subgoal_costs,
    validate_shared_q_lewm_checkpoint,
)


class FakeChunkAgent:
    action_horizon = 5

    def sample_actions(self, observations, goals, seed, temperature):
        assert observations.shape[0] == 1
        assert goals.shape[0] == 1
        assert temperature == 0.0
        return jnp.arange(10, dtype=jnp.float32)[None]


class FakeMultiChunkAgent:
    action_horizon = 5

    def sample_actions(self, observations, goals, seed, temperature):
        count = observations.shape[0]
        blocks = np.tile(np.arange(10, dtype=np.float32), (count, 1))
        blocks[:, 0] += np.arange(count, dtype=np.float32) * 10
        return jnp.asarray(blocks)

    def score_actions(self, observations, goals, actions):
        return actions[:, 0]


class FakeScaler:
    action_dim = 2
    mean = np.array([1.0, -2.0], dtype=np.float32)
    scale = np.array([2.0, 4.0], dtype=np.float32)

    def transform(self, value):
        return (np.asarray(value) - self.mean) / self.scale


def proposal_policy():
    policy = object.__new__(JAXLeWMCEMPolicy)
    policy.horizon = 5
    policy.action_block = 5
    policy.atomic_action_dim = 2
    policy.block_action_dim = 10
    policy.proposal_agent = FakeChunkAgent()
    policy.proposal_temperature = 0.0
    policy.proposal_action_space = 'planner'
    policy.scaler = FakeScaler()
    policy.proposal_num_samples = 1
    policy.proposal_selection = 'mode'
    policy.proposal_elite_size = 1
    policy.proposal_residual_weight = 1.0
    policy.warm_starts = [np.full((2, 10), -3.0, dtype=np.float32)]
    return policy


class ProposalInitializationTest(unittest.TestCase):
    def test_latent_path_selects_k10_token(self):
        self.assertEqual(latent_path_waypoint_index((5, 10), 10), 1)
        with self.assertRaisesRegex(ValueError, 'exactly once'):
            latent_path_waypoint_index((5, 15), 10)

    def test_fixed_subgoal_horizon_maps_k10_to_second_checkpoint(self):
        self.assertEqual(fixed_subgoal_horizon_index(10, 5, 5), 1)
        with self.assertRaisesRegex(ValueError, 'divisible'):
            fixed_subgoal_horizon_index(7, 5, 5)
        with self.assertRaisesRegex(ValueError, 'within'):
            fixed_subgoal_horizon_index(30, 5, 5)

        distances = jnp.asarray([[[5.0, 1.0, 4.0], [2.0, 3.0, 6.0]]])
        np.testing.assert_array_equal(
            select_latent_subgoal_costs(
                distances, 'fixed_subgoal_horizon', fixed_horizon_index=1
            ),
            [1.0, 3.0],
        )

    def test_latent_subgoal_is_held_for_exact_refresh_interval(self):
        policy = object.__new__(JAXLeWMCEMPolicy)
        policy.checkpoint_metadata = {'embed_dim': 3}
        policy._predict_latent_subgoal = lambda current, goal: current + goal
        policy.latent_subgoal_refresh_steps = 5
        policy.latent_subgoals = [None]
        policy.latent_subgoal_ages = np.zeros(1, dtype=np.int64)
        policy.latent_subgoal_generation_counts = np.zeros(1, dtype=np.int64)
        policy.encode_pixels = lambda pixels: np.asarray(
            [[float(np.asarray(pixels).mean())] * 3], dtype=np.float32
        )

        first = policy._planning_target_embedding(
            0,
            np.ones((1, 2, 2, 1), dtype=np.float32),
            np.full((1, 2, 2, 1), 2.0, dtype=np.float32),
        )
        policy.latent_subgoal_ages[0] = 4
        held = policy._planning_target_embedding(
            0,
            np.full((1, 2, 2, 1), 8.0, dtype=np.float32),
            np.full((1, 2, 2, 1), 2.0, dtype=np.float32),
        )
        policy.latent_subgoal_ages[0] = 5
        refreshed = policy._planning_target_embedding(
            0,
            np.full((1, 2, 2, 1), 8.0, dtype=np.float32),
            np.full((1, 2, 2, 1), 2.0, dtype=np.float32),
        )

        np.testing.assert_array_equal(first, [3.0, 3.0, 3.0])
        np.testing.assert_array_equal(held, first)
        np.testing.assert_array_equal(refreshed, [10.0, 10.0, 10.0])
        np.testing.assert_array_equal(policy.latent_subgoal_generation_counts, [2])

    def test_vanilla_planner_uses_dummy_target_without_encoding(self):
        policy = object.__new__(JAXLeWMCEMPolicy)
        policy.checkpoint_metadata = {'embed_dim': 4}
        policy._predict_latent_subgoal = None

        target = policy._planning_target_embedding(0, None, None)

        np.testing.assert_array_equal(target, np.zeros(4, dtype=np.float32))

    def test_flow_subgoal_rng_is_reproducible_per_generation(self):
        policy = object.__new__(JAXLeWMCEMPolicy)
        policy.checkpoint_metadata = {'embed_dim': 3}
        policy.seed = 42
        policy._latent_subgoal_requires_rng = True
        policy._predict_latent_subgoal = lambda current, goal, rng: jax.random.normal(
            rng, current.shape
        )
        policy.latent_subgoal_refresh_steps = 10
        policy.latent_subgoals = [None]
        policy.latent_subgoal_ages = np.zeros(1, dtype=np.int64)
        policy.latent_subgoal_generation_counts = np.zeros(1, dtype=np.int64)
        policy.encode_pixels = lambda pixels: np.zeros((1, 3), dtype=np.float32)

        first = policy._planning_target_embedding(0, np.zeros((1,)), np.zeros((1,)))
        policy.latent_subgoal_ages[0] = 10
        second = policy._planning_target_embedding(0, np.zeros((1,)), np.zeros((1,)))

        self.assertFalse(np.array_equal(first, second))
        replay = object.__new__(JAXLeWMCEMPolicy)
        replay.__dict__.update(policy.__dict__)
        replay.latent_subgoals = [None]
        replay.latent_subgoal_ages = np.zeros(1, dtype=np.int64)
        replay.latent_subgoal_generation_counts = np.zeros(1, dtype=np.int64)
        replay_first = replay._planning_target_embedding(
            0, np.zeros((1,)), np.zeros((1,))
        )
        np.testing.assert_array_equal(first, replay_first)

    def test_checkpoint_path_check_applies_only_to_shared_q(self):
        pi_only = type(
            'PiOnly',
            (),
            {'share_q_encoder': False, 'lewm_checkpoint': '/models/policy.msgpack'},
        )()
        validate_shared_q_lewm_checkpoint('/models/planner.msgpack', pi_only)

        shared_q = type(
            'SharedQ',
            (),
            {'share_q_encoder': True, 'lewm_checkpoint': '/models/policy.msgpack'},
        )()
        with self.assertRaisesRegex(ValueError, 'same normalized LeWM'):
            validate_shared_q_lewm_checkpoint('/models/planner.msgpack', shared_q)

    def test_shared_q_requires_recorded_checkpoint_path(self):
        shared_q = type('SharedQ', (), {'share_q_encoder': True})()
        with self.assertRaisesRegex(ValueError, 'must record'):
            validate_shared_q_lewm_checkpoint('/models/planner.msgpack', shared_q)

    def test_native_q_filter_requires_proposal(self):
        with self.assertRaisesRegex(ValueError, 'requires a proposal agent'):
            JAXLeWMCEMPolicy.__init__(
                object.__new__(JAXLeWMCEMPolicy),
                checkpoint='unused',
                scaler=None,
                seed=0,
                horizon=5,
                receding_horizon=1,
                action_block=5,
                num_samples=300,
                steps=5,
                topk=30,
                var_scale=1.0,
                native_q_keep=150,
            )

    def test_lewm_proposal_selection_requires_actor(self):
        with self.assertRaisesRegex(ValueError, 'requires a proposal agent'):
            JAXLeWMCEMPolicy.__init__(
                object.__new__(JAXLeWMCEMPolicy),
                checkpoint='unused',
                scaler=None,
                seed=0,
                horizon=1,
                receding_horizon=1,
                action_block=5,
                num_samples=300,
                steps=0,
                topk=30,
                var_scale=1.0,
                proposal_num_samples=32,
                proposal_selection='lewm',
            )

    def test_execution_steps_cannot_exceed_selected_blocks(self):
        with self.assertRaisesRegex(ValueError, 'Execution steps'):
            JAXLeWMCEMPolicy.__init__(
                object.__new__(JAXLeWMCEMPolicy),
                checkpoint='unused',
                scaler=None,
                seed=0,
                horizon=1,
                receding_horizon=1,
                action_block=5,
                num_samples=300,
                steps=0,
                topk=30,
                var_scale=1.0,
                execution_steps=6,
            )

    def test_paired_plan_keys_match_with_and_without_proposal(self):
        vanilla = object.__new__(JAXLeWMCEMPolicy)
        guided = object.__new__(JAXLeWMCEMPolicy)
        for policy, proposal_agent in ((vanilla, None), (guided, FakeChunkAgent())):
            policy.seed = 42
            policy.paired_plan_keys = True
            policy.plan_counts = np.zeros(2, dtype=np.int64)
            policy.proposal_agent = proposal_agent

        vanilla_proposal_key, vanilla_plan_key = vanilla._next_plan_keys(1)
        guided_proposal_key, guided_plan_key = guided._next_plan_keys(1)

        np.testing.assert_array_equal(vanilla_plan_key, guided_plan_key)
        np.testing.assert_array_equal(vanilla_proposal_key, guided_proposal_key)

    def test_proposal_replaces_only_first_cem_block(self):
        policy = proposal_policy()
        pixels = np.zeros((1, 16, 16, 3), dtype=np.uint8)
        goals = np.zeros_like(pixels)

        initial = policy._initial_mean(
            0,
            pixels=pixels,
            goals=goals,
            proposal_key=jax.random.PRNGKey(0),
        )

        np.testing.assert_array_equal(initial[0], np.arange(10, dtype=np.float32))
        np.testing.assert_array_equal(initial[1], np.full(10, -3.0, dtype=np.float32))
        np.testing.assert_array_equal(initial[2:], np.zeros((3, 10), dtype=np.float32))

    def test_environment_action_proposal_is_standardized_for_planner(self):
        policy = proposal_policy()
        policy.proposal_action_space = 'environment'
        pixels = np.zeros((1, 16, 16, 3), dtype=np.uint8)

        block = policy._proposal_block(pixels, pixels, jax.random.PRNGKey(0))

        expected = FakeScaler().transform(
            np.arange(10, dtype=np.float32).reshape(5, 2)
        ).reshape(10)
        np.testing.assert_allclose(block, expected)

    def test_lewm_selects_exact_actor_sample_without_cem_averaging(self):
        policy = proposal_policy()
        policy.proposal_agent = FakeMultiChunkAgent()
        policy.proposal_num_samples = 3
        policy.proposal_selection = 'lewm'
        policy._score_plans = lambda pixels, goals, plans: jnp.asarray(
            [2.0, 0.0, 1.0]
        )
        pixels = np.zeros((1, 16, 16, 3), dtype=np.uint8)

        mode, selected = policy._q_selection_blocks(
            pixels, pixels, jax.random.PRNGKey(0)
        )

        np.testing.assert_array_equal(mode, np.arange(10, dtype=np.float32))
        expected = np.arange(10, dtype=np.float32)
        expected[0] = 10.0
        np.testing.assert_array_equal(selected, expected)

    def test_policy_population_cem_blends_elite_mean_toward_mode(self):
        policy = proposal_policy()
        policy.proposal_agent = FakeMultiChunkAgent()
        policy.proposal_num_samples = 4
        policy.proposal_selection = 'lewm_cem'
        policy.proposal_elite_size = 2
        policy.proposal_residual_weight = 0.25
        policy._score_plans = lambda pixels, goals, plans: jnp.asarray(
            [3.0, 0.0, 1.0, 2.0]
        )
        pixels = np.zeros((1, 16, 16, 3), dtype=np.uint8)

        mode, selected = policy._q_selection_blocks(
            pixels, pixels, jax.random.PRNGKey(0)
        )

        np.testing.assert_array_equal(mode, np.arange(10, dtype=np.float32))
        expected = np.arange(10, dtype=np.float32)
        expected[0] = 3.75
        np.testing.assert_allclose(selected, expected)

    def test_native_q_selection_uses_public_score_interface(self):
        policy = proposal_policy()
        policy.proposal_agent = FakeMultiChunkAgent()
        policy.proposal_num_samples = 3
        policy.proposal_selection = 'native_q'
        pixels = np.zeros((1, 16, 16, 3), dtype=np.uint8)

        mode, selected = policy._q_selection_blocks(
            pixels, pixels, jax.random.PRNGKey(0)
        )

        np.testing.assert_array_equal(mode, np.arange(10, dtype=np.float32))
        expected = np.arange(10, dtype=np.float32)
        expected[0] = 20.0
        np.testing.assert_array_equal(selected, expected)

    def test_proposal_shape_is_checked(self):
        policy = proposal_policy()
        policy.block_action_dim = 9
        pixels = np.zeros((1, 16, 16, 3), dtype=np.uint8)

        with self.assertRaisesRegex(ValueError, 'Proposal returned'):
            policy._proposal_block(pixels, pixels, jax.random.PRNGKey(0))


if __name__ == '__main__':
    unittest.main()
