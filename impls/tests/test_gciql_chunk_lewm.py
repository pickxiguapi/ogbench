import unittest

import jax
import jax.numpy as jnp

from agents.gciql_chunk_lewm import LeWMGCIQLChunkAgent, get_config


class LeWMGCIQLChunkAgentTest(unittest.TestCase):
    def make_agent_and_batch(self, shared):
        config = get_config()
        config.actor_hidden_dims = (8, 8)
        config.value_hidden_dims = (8, 8)
        config.chunk_size = 2
        config.latent_dim = 6
        config.encoder = 'impala_debug'
        config.share_q_encoder = 'q' in shared
        config.share_v_encoder = 'v' in shared
        config.share_pi_encoder = 'pi' in shared

        pixels = jnp.zeros((2, 16, 16, 3), dtype=jnp.uint8)
        latents = jnp.zeros((2, config.latent_dim), dtype=jnp.float32)
        actions = jnp.zeros((2, 4), dtype=jnp.float32)
        agent = LeWMGCIQLChunkAgent.create(
            0, pixels[:1], latents[:1], actions[:1], config
        )
        batch = {
            'observations': pixels,
            'next_observations': pixels,
            'value_goals': pixels,
            'actor_goals': pixels,
            'lewm_observations': latents,
            'lewm_next_observations': latents,
            'lewm_value_goals': latents,
            'lewm_actor_goals': latents,
            'actions': actions,
            'rewards': -jnp.ones((2,), dtype=jnp.float32),
            'masks': jnp.ones((2,), dtype=jnp.float32),
        }
        return agent, batch

    def test_requested_sharing_variants_have_finite_losses(self):
        for shared in ({'pi'}, {'q', 'v', 'pi'}, {'q', 'v'}):
            with self.subTest(shared=shared):
                agent, batch = self.make_agent_and_batch(shared)
                loss, info = agent.total_loss(batch, grad_params=None)
                self.assertTrue(jnp.isfinite(loss))
                self.assertIn('actor/actor_loss', info)

    def test_all_shared_agent_updates_and_emits_chunk(self):
        agent, batch = self.make_agent_and_batch({'q', 'v', 'pi'})
        agent, info = agent.update(batch)
        actions = agent.sample_actions(
            batch['lewm_observations'][:1],
            batch['lewm_actor_goals'][:1],
            seed=agent.rng,
            temperature=0.0,
        )
        self.assertEqual(actions.shape, (1, 4))
        self.assertTrue(jnp.isfinite(info['actor/actor_loss']))

    def test_lewm_inputs_are_required_only_by_shared_modules(self):
        agent, batch = self.make_agent_and_batch({'pi'})
        del batch['lewm_value_goals']
        # Q and V are pixel modules, so actor-goal LeWM inputs are sufficient.
        loss, _ = agent.total_loss(batch, grad_params=None)
        self.assertTrue(jnp.isfinite(loss))


if __name__ == '__main__':
    unittest.main()
