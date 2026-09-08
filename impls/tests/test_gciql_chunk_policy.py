import unittest

import jax.numpy as jnp
import numpy as np

from gciql_chunk_policy import PlannerBlockPolicyAdapter


class FakePolicy:
    def __init__(self, action_horizon, actions):
        self.action_horizon = action_horizon
        self.actions = jnp.asarray(actions, dtype=jnp.float32)

    def sample_actions(self, **kwargs):
        del kwargs
        return self.actions


class PlannerBlockPolicyAdapterTest(unittest.TestCase):
    def test_zero_pads_short_policy_prefix(self):
        policy = FakePolicy(1, [[1.0, 2.0]])
        adapter = PlannerBlockPolicyAdapter(policy, 5)

        actual = np.asarray(adapter.sample_actions())

        np.testing.assert_array_equal(
            actual,
            [[1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
        )

    def test_truncates_long_policy_prefix(self):
        policy = FakePolicy(10, [list(range(20))])
        adapter = PlannerBlockPolicyAdapter(policy, 5)

        actual = np.asarray(adapter.sample_actions())

        np.testing.assert_array_equal(actual, [list(range(10))])

    def test_rejects_malformed_policy_width(self):
        policy = FakePolicy(3, [[1.0, 2.0, 3.0, 4.0]])
        adapter = PlannerBlockPolicyAdapter(policy, 5)

        with self.assertRaisesRegex(ValueError, 'not divisible'):
            adapter.sample_actions()


if __name__ == '__main__':
    unittest.main()
