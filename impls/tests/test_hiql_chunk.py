import unittest

import jax.numpy as jnp
import numpy as np

from agents.hiql import get_config as get_hiql_config
from agents.hiql_chunk import HIQLChunkAgent, get_config
from utils.datasets import Dataset, HIQLChunkDataset


def make_config(chunk_size=3):
    config = get_config()
    config.chunk_size = chunk_size
    config.frame_stack = None
    config.p_aug = 0.0
    return config


def make_dataset(chunk_size=3, terminals=None, valids=None, observation_dim=1):
    size = 10
    if terminals is None:
        terminals = np.asarray([0] * (size - 1) + [1], dtype=np.float32)
    if valids is None:
        valids = 1.0 - terminals
    base = Dataset.create(
        freeze=False,
        observations=np.repeat(np.arange(size, dtype=np.float32)[:, None], observation_dim, axis=1),
        actions=np.arange(size * 2, dtype=np.float32).reshape(size, 2),
        terminals=terminals,
        valids=valids,
    )
    return HIQLChunkDataset(base, make_config(chunk_size))


class HIQLChunkDatasetTest(unittest.TestCase):
    def test_only_low_level_transition_is_chunked(self):
        dataset = make_dataset(chunk_size=3)
        np.random.seed(0)
        batch = dataset.sample(1, idxs=np.asarray([1]), evaluation=True)

        np.testing.assert_array_equal(batch['observations'], [[1.0]])
        np.testing.assert_array_equal(batch['next_observations'], [[2.0]])
        np.testing.assert_array_equal(batch['low_actor_next_observations'], [[4.0]])
        np.testing.assert_array_equal(batch['actions'], [[2.0, 3.0, 4.0, 5.0, 6.0, 7.0]])
        np.testing.assert_allclose(batch['low_value_rewards'], [-2.9701])
        np.testing.assert_array_equal(batch['low_value_masks'], [1.0])

    def test_chunk_cannot_cross_episode_boundary(self):
        terminals = np.asarray([0, 0, 0, 1, 0, 0, 0, 0, 0, 1], dtype=np.float32)
        valids = 1.0 - terminals
        dataset = make_dataset(chunk_size=3, terminals=terminals, valids=valids)

        with self.assertRaisesRegex(ValueError, 'complete, valid action chunks'):
            dataset.sample(1, idxs=np.asarray([2]), evaluation=True)

    def test_k_one_keeps_atomic_value_and_low_level_next_state_aligned(self):
        dataset = make_dataset(chunk_size=1)
        np.random.seed(0)
        batch = dataset.sample(1, idxs=np.asarray([2]), evaluation=True)

        np.testing.assert_array_equal(batch['next_observations'], [[3.0]])
        np.testing.assert_array_equal(batch['low_actor_next_observations'], [[3.0]])
        np.testing.assert_array_equal(batch['actions'], [[4.0, 5.0]])


class HIQLChunkAgentTest(unittest.TestCase):
    def make_agent(self, chunk_size=3):
        config = get_config()
        config.encoder = None
        config.chunk_size = chunk_size
        config.actor_hidden_dims = (16, 16)
        config.value_hidden_dims = (16, 16)
        observations = jnp.zeros((2, 4), dtype=jnp.float32)
        actions = jnp.zeros((2, chunk_size * 2), dtype=jnp.float32)
        return HIQLChunkAgent.create(0, observations, actions, config), config

    def test_default_config_changes_only_chunk_specific_fields(self):
        base_config = get_hiql_config()
        config = get_config()

        self.assertEqual(config.agent_name, 'hiql_chunk')
        self.assertEqual(config.dataset_class, 'HIQLChunkDataset')
        self.assertEqual(config.chunk_size, 5)
        self.assertFalse(config.discrete)
        self.assertEqual(set(config.keys()), set(base_config.keys()) | {'chunk_size'})
        for key in base_config:
            if key not in {'agent_name', 'dataset_class'}:
                self.assertEqual(config[key], base_config[key], key)

    def test_agent_has_independent_high_and_low_values_and_targets(self):
        agent, _ = self.make_agent(chunk_size=3)
        param_names = set(agent.network.params)
        self.assertIn('modules_high_value', param_names)
        self.assertIn('modules_target_high_value', param_names)
        self.assertIn('modules_low_value', param_names)
        self.assertIn('modules_target_low_value', param_names)
        self.assertNotEqual(
            id(agent.network.params['modules_high_value']),
            id(agent.network.params['modules_low_value']),
        )

    def test_agent_emits_flattened_chunk_and_declares_horizon(self):
        agent, _ = self.make_agent(chunk_size=3)
        observations = jnp.zeros((1, 4), dtype=jnp.float32)
        sampled_actions = agent.sample_actions(observations, observations, seed=agent.rng)

        self.assertEqual(agent.action_horizon, 3)
        self.assertEqual(sampled_actions.shape, (1, 6))

    def test_low_actor_advantage_uses_low_value_and_chunk_next_observation(self):
        agent, _ = self.make_agent(chunk_size=3)
        observations = jnp.zeros((2, 4), dtype=jnp.float32)
        chunk_next = jnp.ones_like(observations)
        goals = jnp.full_like(observations, 2.0)
        actions = jnp.zeros((2, 6), dtype=jnp.float32)
        batch = {
            'observations': observations,
            'next_observations': jnp.full_like(observations, 99.0),
            'low_actor_next_observations': chunk_next,
            'low_actor_goals': goals,
            'actions': actions,
        }

        _, info = agent.low_actor_loss(batch, grad_params=None)
        v1, v2 = agent.network.select('low_value')(observations, goals)
        nv1, nv2 = agent.network.select('low_value')(chunk_next, goals)
        expected_adv = (((nv1 + nv2) - (v1 + v2)) / 2).mean()

        np.testing.assert_allclose(info['adv'], expected_adv, rtol=1e-6)

    def test_full_update_reports_both_value_levels(self):
        agent, config = self.make_agent(chunk_size=3)
        dataset = make_dataset(chunk_size=3, observation_dim=4)
        config.frame_stack = None
        batch = dataset.sample(2, idxs=np.asarray([0, 1]), evaluation=True)
        batch = {key: jnp.asarray(value) for key, value in batch.items()}

        updated_agent, info = agent.update(batch)

        self.assertEqual(int(updated_agent.network.step), int(agent.network.step) + 1)
        self.assertIn('high_value/value_loss', info)
        self.assertIn('low_value/value_loss', info)

    def test_impala_small_visual_update_smoke(self):
        config = get_config()
        config.encoder = 'impala_small'
        config.chunk_size = 5
        config.actor_hidden_dims = (32, 32)
        config.value_hidden_dims = (32, 32)
        observations = jnp.zeros((2, 32, 32, 3), dtype=jnp.uint8)
        goals = jnp.full_like(observations, 127)
        actions = jnp.zeros((2, 10), dtype=jnp.float32)
        agent = HIQLChunkAgent.create(0, observations, actions, config)
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
        self.assertTrue(np.isfinite(np.asarray(info['high_value/value_loss'])))
        self.assertTrue(np.isfinite(np.asarray(info['low_value/value_loss'])))


if __name__ == '__main__':
    unittest.main()
