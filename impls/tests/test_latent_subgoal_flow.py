import unittest

import jax
import jax.numpy as jnp
import numpy as np
from latent_subgoal import (
    LatentPathFlow,
    LatentSubgoalMLP,
    LatentSubgoalFlowTransformer,
    latent_path_waypoint_steps,
    sample_conditional_flow,
    sample_conditional_flow_candidates,
    sample_conditional_path_flow,
    sample_conditional_path_flow_candidates,
    select_latent_medoid,
    select_latent_path_medoid,
    sinusoidal_time_embedding,
)


class LatentSubgoalFlowTest(unittest.TestCase):
    def test_parameter_matched_history_mlp_and_endpoint_flow(self):
        history = jnp.zeros((1, 3, 192), dtype=jnp.float32)
        goal = jnp.zeros((1, 192), dtype=jnp.float32)
        flow_time = jnp.zeros((1,), dtype=jnp.float32)
        key = jax.random.PRNGKey(0)

        mlp = LatentSubgoalMLP(
            embed_dim=192, hidden_dims=(2048, 2048, 2048, 2048, 2048)
        )
        mlp_variables = mlp.init(key, history, goal)
        endpoint_flow = LatentPathFlow(
            embed_dim=192,
            num_waypoints=1,
            hidden_dim=512,
            depth=4,
            num_heads=8,
            ff_dim=2048,
            time_dim=64,
            history_size=3,
        )
        endpoint_variables = endpoint_flow.init(
            key,
            jnp.zeros((1, 1, 192), dtype=jnp.float32),
            history,
            goal,
            flow_time,
        )
        path_flow = LatentPathFlow(embed_dim=192, history_size=3)
        path_variables = path_flow.init(
            key,
            jnp.zeros((1, 2, 192), dtype=jnp.float32),
            history,
            goal,
            flow_time,
        )

        counts = [
            sum(value.size for value in jax.tree_util.tree_leaves(tree['params']))
            for tree in (mlp_variables, endpoint_variables, path_variables)
        ]
        self.assertEqual(counts, [18_774_208, 18_742_464, 18_742_976])
        self.assertLess((max(counts) - min(counts)) / max(counts), 0.01)

    def test_waypoints_are_derived_from_subgoal_steps_and_action_block(self):
        self.assertEqual(latent_path_waypoint_steps(10, 5), (5, 10))
        self.assertEqual(latent_path_waypoint_steps(15, 5), (5, 10, 15))
        with self.assertRaisesRegex(ValueError, 'divisible'):
            latent_path_waypoint_steps(10, 4)

    def test_flow_time_embedding_defaults_to_unscaled_unit_interval(self):
        embedding = sinusoidal_time_embedding(jnp.asarray([1.0]), dim=4)
        expected = jnp.asarray(
            [[jnp.sin(1.0), jnp.sin(1e-4), jnp.cos(1.0), jnp.cos(1e-4)]]
        )
        np.testing.assert_allclose(embedding, expected, rtol=1e-6, atol=1e-6)

    def test_latent_path_flow_matches_leflow_parameter_budget_and_shape(self):
        model = LatentPathFlow(embed_dim=192, history_size=3)
        current = jnp.zeros((2, 3, 192), dtype=jnp.float32)
        goal = jnp.zeros((2, 192), dtype=jnp.float32)
        path = jnp.zeros((2, 2, 192), dtype=jnp.float32)
        variables = model.init(
            jax.random.PRNGKey(0),
            path,
            current,
            goal,
            jnp.zeros((2,), dtype=jnp.float32),
        )
        output = model.apply(
            variables, path, current, goal, jnp.zeros((2,), dtype=jnp.float32)
        )
        parameter_count = sum(
            value.size for value in jax.tree_util.tree_leaves(variables['params'])
        )

        self.assertEqual(output.shape, (2, 2, 192))
        self.assertGreaterEqual(parameter_count, 10_000_000)
        self.assertLessEqual(parameter_count, 20_000_000)

    def test_history_conditioned_path_flow_uses_adaln_conditions(self):
        model = LatentPathFlow(
            embed_dim=8,
            num_waypoints=2,
            hidden_dim=16,
            depth=2,
            num_heads=4,
            ff_dim=32,
            time_dim=8,
            history_size=3,
        )
        path = jnp.zeros((2, 2, 8), dtype=jnp.float32)
        history = jnp.zeros((2, 3, 8), dtype=jnp.float32)
        goal = jnp.ones((2, 8), dtype=jnp.float32)
        variables = model.init(
            jax.random.PRNGKey(0),
            path,
            history,
            goal,
            jnp.zeros((2,), dtype=jnp.float32),
        )
        baseline = model.apply(
            variables, path, history, goal, jnp.zeros((2,), dtype=jnp.float32)
        )
        changed_history = history.at[:, 0].set(1.0)
        changed = model.apply(
            variables,
            path,
            changed_history,
            goal,
            jnp.zeros((2,), dtype=jnp.float32),
        )
        changed_goal = model.apply(
            variables,
            path,
            history,
            goal + 1.0,
            jnp.zeros((2,), dtype=jnp.float32),
        )

        self.assertEqual(baseline.shape, (2, 2, 8))
        self.assertFalse(np.array_equal(np.asarray(baseline), np.asarray(changed)))
        self.assertFalse(
            np.array_equal(np.asarray(baseline), np.asarray(changed_goal))
        )
        self.assertIn(
            'condition_modulation', variables['params']['encoder_block_0']
        )

    def test_latent_path_flow_euler_sampling_is_deterministic(self):
        model = LatentPathFlow(
            embed_dim=8,
            num_waypoints=2,
            hidden_dim=16,
            depth=2,
            num_heads=4,
            ff_dim=32,
            time_dim=8,
        )
        current = jnp.ones((2, 8), dtype=jnp.float32)
        goal = jnp.full((2, 8), 2.0, dtype=jnp.float32)
        path = jnp.zeros((2, 2, 8), dtype=jnp.float32)
        variables = model.init(
            jax.random.PRNGKey(0),
            path,
            current,
            goal,
            jnp.zeros((2,), dtype=jnp.float32),
        )
        first = sample_conditional_path_flow(
            model,
            variables['params'],
            current,
            goal,
            jax.random.PRNGKey(7),
            num_steps=4,
            solver='euler',
        )
        repeated = sample_conditional_path_flow(
            model,
            variables['params'],
            current,
            goal,
            jax.random.PRNGKey(7),
            num_steps=4,
            solver='euler',
        )

        self.assertEqual(first.shape, (2, 2, 8))
        self.assertTrue(np.isfinite(np.asarray(first)).all())
        np.testing.assert_array_equal(first, repeated)

    def test_latent_path_flow_can_draw_multiple_candidates(self):
        from train_latent_subgoal_gcbc import make_predict_indices

        model = LatentPathFlow(
            embed_dim=8,
            num_waypoints=2,
            hidden_dim=16,
            depth=1,
            num_heads=4,
            ff_dim=32,
            time_dim=8,
            history_size=3,
        )
        current = jnp.ones((2, 3, 8), dtype=jnp.float32)
        goal = jnp.full((2, 8), 2.0, dtype=jnp.float32)
        variables = model.init(
            jax.random.PRNGKey(0),
            jnp.zeros((2, 2, 8), dtype=jnp.float32),
            current,
            goal,
            jnp.zeros((2,), dtype=jnp.float32),
        )
        candidates = sample_conditional_path_flow_candidates(
            model,
            variables['params'],
            current,
            goal,
            jax.random.PRNGKey(7),
            num_samples=4,
            num_steps=2,
            solver='euler',
        )
        repeated = sample_conditional_path_flow_candidates(
            model,
            variables['params'],
            current,
            goal,
            jax.random.PRNGKey(7),
            num_samples=4,
            num_steps=2,
            solver='euler',
        )

        self.assertEqual(candidates.shape, (2, 4, 2, 8))
        np.testing.assert_array_equal(candidates, repeated)
        self.assertFalse(
            np.array_equal(np.asarray(candidates[:, 0]), np.asarray(candidates[:, 1]))
        )
        z = jnp.concatenate((current.reshape(6, 8), goal), axis=0)
        current_indices = jnp.asarray([2, 5], dtype=jnp.int32)
        history_indices = jnp.asarray([[0, 1, 2], [3, 4, 5]], dtype=jnp.int32)
        goal_indices = jnp.asarray([6, 7], dtype=jnp.int32)
        validation_predict = make_predict_indices(
            model,
            path_flow_matching=True,
            history_size=3,
            flow_sampling_steps=2,
            flow_solver='euler',
            num_samples=4,
        )
        validation_prediction = validation_predict(
            variables['params'],
            z,
            current_indices,
            history_indices,
            goal_indices,
            jax.random.PRNGKey(7),
        )
        np.testing.assert_array_equal(
            validation_prediction, select_latent_path_medoid(candidates)
        )

    def test_path_medoid_returns_an_actual_central_sample(self):
        candidates = jnp.asarray(
            [[[[0.0]], [[1.0]], [[10.0]]]], dtype=jnp.float32
        )

        selected = select_latent_path_medoid(candidates)

        self.assertEqual(selected.shape, (1, 1, 1))
        np.testing.assert_array_equal(selected, [[[1.0]]])

    def test_formal_transformer_is_within_parameter_budget(self):
        model = LatentSubgoalFlowTransformer(
            embed_dim=192,
            model_dim=384,
            num_layers=8,
            num_heads=8,
            mlp_dim=1536,
        )
        latents = jnp.zeros((1, 192), dtype=jnp.float32)
        variables = model.init(
            jax.random.PRNGKey(0),
            latents,
            latents,
            latents,
            jnp.zeros((1,), dtype=jnp.float32),
        )
        parameter_count = sum(
            value.size for value in jax.tree_util.tree_leaves(variables['params'])
        )

        self.assertGreaterEqual(parameter_count, 10_000_000)
        self.assertLessEqual(parameter_count, 20_000_000)

    def test_flow_sampling_is_finite_and_key_deterministic(self):
        model = LatentSubgoalFlowTransformer(
            embed_dim=8,
            model_dim=16,
            num_layers=2,
            num_heads=4,
            mlp_dim=32,
        )
        current = jnp.ones((2, 8), dtype=jnp.float32)
        goal = jnp.full((2, 8), 2.0, dtype=jnp.float32)
        variables = model.init(
            jax.random.PRNGKey(0),
            current,
            current,
            goal,
            jnp.zeros((2,), dtype=jnp.float32),
        )

        first = sample_conditional_flow(
            model,
            variables['params'],
            current,
            goal,
            jax.random.PRNGKey(7),
            num_steps=4,
            solver='heun',
        )
        repeated = sample_conditional_flow(
            model,
            variables['params'],
            current,
            goal,
            jax.random.PRNGKey(7),
            num_steps=4,
            solver='heun',
        )
        different = sample_conditional_flow(
            model,
            variables['params'],
            current,
            goal,
            jax.random.PRNGKey(8),
            num_steps=4,
            solver='heun',
        )

        self.assertEqual(first.shape, current.shape)
        self.assertTrue(np.isfinite(np.asarray(first)).all())
        np.testing.assert_array_equal(first, repeated)
        self.assertFalse(np.array_equal(np.asarray(first), np.asarray(different)))

    def test_transformer_flow_can_draw_multiple_candidates(self):
        from train_latent_subgoal_gcbc import make_predict_indices

        model = LatentSubgoalFlowTransformer(
            embed_dim=8,
            model_dim=16,
            num_layers=1,
            num_heads=4,
            mlp_dim=32,
        )
        current = jnp.ones((2, 8), dtype=jnp.float32)
        goal = jnp.full((2, 8), 2.0, dtype=jnp.float32)
        variables = model.init(
            jax.random.PRNGKey(0),
            current,
            current,
            goal,
            jnp.zeros((2,), dtype=jnp.float32),
        )

        candidates = sample_conditional_flow_candidates(
            model,
            variables['params'],
            current,
            goal,
            jax.random.PRNGKey(7),
            num_samples=4,
            num_steps=2,
            solver='euler',
        )
        repeated = sample_conditional_flow_candidates(
            model,
            variables['params'],
            current,
            goal,
            jax.random.PRNGKey(7),
            num_samples=4,
            num_steps=2,
            solver='euler',
        )

        self.assertEqual(candidates.shape, (2, 4, 8))
        np.testing.assert_array_equal(candidates, repeated)
        self.assertFalse(
            np.array_equal(np.asarray(candidates[:, 0]), np.asarray(candidates[:, 1]))
        )
        z = jnp.concatenate((current, goal), axis=0)
        current_indices = jnp.asarray([0, 1], dtype=jnp.int32)
        goal_indices = jnp.asarray([2, 3], dtype=jnp.int32)
        validation_predict = make_predict_indices(
            model,
            flow_matching=True,
            flow_sampling_steps=2,
            flow_solver='euler',
            num_samples=4,
        )
        validation_prediction = validation_predict(
            variables['params'],
            z,
            current_indices,
            jnp.zeros((2, 1), dtype=jnp.int32),
            goal_indices,
            jax.random.PRNGKey(7),
        )
        np.testing.assert_array_equal(
            validation_prediction, select_latent_medoid(candidates)
        )

    def test_latent_medoid_returns_an_actual_central_sample(self):
        candidates = jnp.asarray(
            [[[0.0], [1.0], [10.0]]], dtype=jnp.float32
        )

        selected = select_latent_medoid(candidates)

        self.assertEqual(selected.shape, (1, 1))
        np.testing.assert_array_equal(selected, [[1.0]])


if __name__ == '__main__':
    unittest.main()
