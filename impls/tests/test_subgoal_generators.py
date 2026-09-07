import unittest

import jax
import jax.numpy as jnp
import numpy as np
from subgoal_generators import (
    LatentPathFlow,
    LatentSubgoalMLP,
    sample_path,
    sample_path_candidates,
    select_path_medoid,
    waypoint_steps,
)


class SubgoalGeneratorTest(unittest.TestCase):
    def setUp(self):
        self.model = LatentPathFlow(
            embed_dim=8,
            num_waypoints=2,
            hidden_dim=16,
            depth=2,
            num_heads=4,
            ff_dim=32,
            time_dim=8,
            history_size=3,
        )
        self.history = jnp.zeros((2, 3, 8), dtype=jnp.float32)
        self.goals = jnp.ones((2, 8), dtype=jnp.float32)
        self.params = self.model.init(
            jax.random.PRNGKey(0),
            jnp.zeros((2, 2, 8), dtype=jnp.float32),
            self.history,
            self.goals,
            jnp.zeros((2,), dtype=jnp.float32),
        )['params']

    def test_waypoints_follow_action_chunks(self):
        self.assertEqual(waypoint_steps(10, 5), (5, 10))
        with self.assertRaisesRegex(ValueError, 'divisible'):
            waypoint_steps(9, 5)

    def test_sampling_is_deterministic_for_a_fixed_key(self):
        key = jax.random.PRNGKey(7)
        first = sample_path(self.model, self.params, self.history, self.goals, key, num_steps=2)
        second = sample_path(self.model, self.params, self.history, self.goals, key, num_steps=2)
        self.assertEqual(first.shape, (2, 2, 8))
        np.testing.assert_allclose(first, second)

    def test_candidate_and_medoid_shapes(self):
        candidates = sample_path_candidates(
            self.model,
            self.params,
            self.history,
            self.goals,
            jax.random.PRNGKey(3),
            num_samples=3,
            num_steps=1,
        )
        self.assertEqual(candidates.shape, (2, 3, 2, 8))
        self.assertEqual(select_path_medoid(candidates).shape, (2, 2, 8))

    def test_endpoint_flow_is_the_one_waypoint_configuration(self):
        model = LatentPathFlow(
            embed_dim=8,
            num_waypoints=1,
            hidden_dim=16,
            depth=1,
            num_heads=4,
            ff_dim=32,
            time_dim=8,
            history_size=3,
        )
        variables = model.init(
            jax.random.PRNGKey(4),
            jnp.zeros((2, 1, 8)),
            self.history,
            self.goals,
            jnp.zeros((2,)),
        )
        output = sample_path(model, variables['params'], self.history, self.goals, jax.random.PRNGKey(5), num_steps=2)
        self.assertEqual(output.shape, (2, 1, 8))

    def test_mlp_predicts_one_endpoint_from_history_and_goal(self):
        model = LatentSubgoalMLP(embed_dim=8, hidden_dims=(16, 16))
        variables = model.init(jax.random.PRNGKey(6), self.history, self.goals)
        output = model.apply(variables, self.history, self.goals)
        self.assertEqual(output.shape, (2, 8))


if __name__ == '__main__':
    unittest.main()
