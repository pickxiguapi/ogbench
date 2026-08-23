import unittest

import jax.numpy as jnp

from agents.gciql_chunk_lewm_shared import (
    LeWMSharedGCIQLChunkEvaluator,
    get_config,
)


class SharedEvaluatorTest(unittest.TestCase):
    def test_update_and_score_shapes(self):
        config = get_config()
        config.latent_dim = 8
        config.value_hidden_dims = (16, 16)
        evaluator = LeWMSharedGCIQLChunkEvaluator.create(
            0,
            jnp.zeros((1, 8), dtype=jnp.float32),
            jnp.zeros((1, 10), dtype=jnp.float32),
            config,
        )
        batch = {
            'observations': jnp.zeros((4, 8), dtype=jnp.float32),
            'next_observations': jnp.ones((4, 8), dtype=jnp.float32),
            'goals': jnp.ones((4, 8), dtype=jnp.float32),
            'actions': jnp.zeros((4, 10), dtype=jnp.float32),
            'rewards': -jnp.ones((4,), dtype=jnp.float32),
            'masks': jnp.ones((4,), dtype=jnp.float32),
        }
        evaluator, info = evaluator.update(batch)
        scores = evaluator.score_actions(
            batch['observations'], batch['goals'], batch['actions']
        )
        self.assertEqual(scores.shape, (4,))
        self.assertTrue(jnp.isfinite(info['loss']))


if __name__ == '__main__':
    unittest.main()
