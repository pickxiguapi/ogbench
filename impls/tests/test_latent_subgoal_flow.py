import unittest

import jax
import jax.numpy as jnp
import numpy as np
from latent_subgoal import LatentSubgoalFlowTransformer, sample_conditional_flow


class LatentSubgoalFlowTest(unittest.TestCase):
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


if __name__ == '__main__':
    unittest.main()
