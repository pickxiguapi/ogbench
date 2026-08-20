import unittest

import jax.numpy as jnp
import numpy as np

from agents.hiql_chunk_share_v import HIQLChunkShareVAgent, get_config


class HIQLChunkShareVAgentTest(unittest.TestCase):
    def make_agent(self, chunk_size=3):
        config = get_config()
        config.encoder = None
        config.chunk_size = chunk_size
        config.actor_hidden_dims = (16, 16)
        config.value_hidden_dims = (16, 16)
        observations = jnp.zeros((2, 4), dtype=jnp.float32)
        actions = jnp.zeros((2, chunk_size * 2), dtype=jnp.float32)
        return HIQLChunkShareVAgent.create(0, observations, actions, config), config

    def test_network_has_one_shared_value_and_no_low_value(self):
        agent, _ = self.make_agent()
        names = set(agent.network.params)

        self.assertIn('modules_value', names)
        self.assertIn('modules_target_value', names)
        self.assertIn('modules_low_critic', names)
        self.assertNotIn('modules_low_value', names)
        self.assertNotIn('modules_high_value', names)

    def test_low_actor_advantage_is_chunk_q_minus_shared_value(self):
        agent, _ = self.make_agent()
        observations = jnp.zeros((2, 4), dtype=jnp.float32)
        goals = jnp.full_like(observations, 2.0)
        actions = jnp.zeros((2, 6), dtype=jnp.float32)
        batch = {'observations': observations, 'low_actor_goals': goals, 'actions': actions}

        _, info = agent.low_actor_loss(batch, grad_params=None)
        values, goal_reps = agent.network.select('value')(
            observations, goals, return_goal_rep=True
        )
        q1, q2 = agent.network.select('low_critic')(observations, goal_reps, actions)
        expected_adv = (jnp.minimum(q1, q2) - values.mean(axis=0)).mean()

        np.testing.assert_allclose(info['adv'], expected_adv, rtol=1e-6)

    def test_low_critic_bootstraps_from_shared_target_value(self):
        agent, config = self.make_agent()
        observations = jnp.zeros((2, 4), dtype=jnp.float32)
        chunk_next = jnp.ones_like(observations)
        goals = jnp.full_like(observations, 2.0)
        actions = jnp.zeros((2, 6), dtype=jnp.float32)
        batch = {
            'observations': observations,
            'low_actor_next_observations': chunk_next,
            'low_actor_goals': goals,
            'actions': actions,
            'low_value_rewards': jnp.asarray([-2.0, -1.0]),
            'low_value_masks': jnp.asarray([1.0, 0.0]),
        }

        loss, info = agent.low_critic_loss(batch, grad_params=None)
        next_v1, next_v2 = agent.network.select('target_value')(chunk_next, goals)
        target_q = (
            batch['low_value_rewards']
            + config.discount**config.chunk_size
            * batch['low_value_masks']
            * jnp.minimum(next_v1, next_v2)
        )
        _, goal_reps = agent.network.select('value')(
            observations, goals, return_goal_rep=True
        )
        q1, q2 = agent.network.select('low_critic')(observations, goal_reps, actions)
        expected_loss = ((q1 - target_q) ** 2 + (q2 - target_q) ** 2).mean()

        np.testing.assert_allclose(loss, expected_loss, rtol=1e-6)
        np.testing.assert_allclose(info['q_mean'], target_q.mean(), rtol=1e-6)

    def test_agent_emits_flattened_chunk(self):
        agent, _ = self.make_agent()
        observations = jnp.zeros((1, 4), dtype=jnp.float32)
        actions = agent.sample_actions(observations, observations, seed=agent.rng)

        self.assertEqual(agent.action_horizon, 3)
        self.assertEqual(actions.shape, (1, 6))

    def test_impala_visual_update_smoke(self):
        config = get_config()
        config.encoder = 'impala_small'
        config.chunk_size = 5
        config.actor_hidden_dims = (32, 32)
        config.value_hidden_dims = (32, 32)
        observations = jnp.zeros((2, 32, 32, 3), dtype=jnp.uint8)
        goals = jnp.full_like(observations, 127)
        actions = jnp.zeros((2, 10), dtype=jnp.float32)
        agent = HIQLChunkShareVAgent.create(0, observations, actions, config)
        batch = {
            'observations': observations,
            'next_observations': jnp.ones_like(observations),
            'value_goals': goals,
            'rewards': jnp.asarray([-1.0, -1.0]),
            'masks': jnp.ones(2),
            'low_actor_next_observations': jnp.full_like(observations, 2),
            'low_actor_goals': goals,
            'low_value_rewards': jnp.asarray([-4.900995, -4.900995]),
            'low_value_masks': jnp.ones(2),
            'high_actor_goals': goals,
            'high_actor_targets': jnp.full_like(observations, 3),
            'actions': actions,
        }

        updated_agent, info = agent.update(batch)

        self.assertEqual(int(updated_agent.network.step), int(agent.network.step) + 1)
        self.assertTrue(np.isfinite(np.asarray(info['value/value_loss'])))
        self.assertTrue(np.isfinite(np.asarray(info['low_critic/critic_loss'])))


if __name__ == '__main__':
    unittest.main()
