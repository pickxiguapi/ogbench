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


def proposal_policy():
    policy = object.__new__(JAXLeWMCEMPolicy)
    policy.horizon = 5
    policy.action_block = 5
    policy.atomic_action_dim = 2
    policy.block_action_dim = 10
    policy.proposal_agent = FakeChunkAgent()
    policy.proposal_temperature = 0.0
    policy.warm_starts = [np.full((2, 10), -3.0, dtype=np.float32)]
    return policy


class ProposalInitializationTest(unittest.TestCase):
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

    def test_proposal_shape_is_checked(self):
        policy = proposal_policy()
        policy.block_action_dim = 9
        pixels = np.zeros((1, 16, 16, 3), dtype=np.uint8)

        with self.assertRaisesRegex(ValueError, 'Proposal returned'):
            policy._proposal_block(pixels, pixels, jax.random.PRNGKey(0))


if __name__ == '__main__':
    unittest.main()
