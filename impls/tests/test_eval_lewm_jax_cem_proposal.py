import jax
import jax.numpy as jnp
import numpy as np
import unittest

from eval_lewm_jax_cem import JAXLeWMCEMPolicy


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
    policy.shared_q_evaluator = None
    policy.warm_starts = [np.full((2, 10), -3.0, dtype=np.float32)]
    return policy


class ProposalInitializationTest(unittest.TestCase):
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

    def test_proposal_shape_is_checked(self):
        policy = proposal_policy()
        policy.block_action_dim = 9
        pixels = np.zeros((1, 16, 16, 3), dtype=np.uint8)

        with self.assertRaisesRegex(ValueError, 'Proposal returned'):
            policy._proposal_block(pixels, pixels, jax.random.PRNGKey(0))


if __name__ == '__main__':
    unittest.main()
