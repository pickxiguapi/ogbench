import unittest

import jax.numpy as jnp
import numpy as np
from agents.gciql import GCIQLAgent
from agents.gciql import get_config as get_gciql_config
from agents.gciql_chunk import GCIQLChunkAgent, get_config
from utils.datasets import Dataset, GCChunkDataset


class FixedGoalGCChunkDataset(GCChunkDataset):
    """GCChunkDataset with deterministic value and actor goals for tests."""

    def set_goals(self, value_goal_idxs, actor_goal_idxs):
        self._test_goals = [np.asarray(value_goal_idxs), np.asarray(actor_goal_idxs)]

    def sample_goals(self, *args, **kwargs):
        return self._test_goals.pop(0)


def make_config(chunk_size=3, discount=0.9):
    config = get_config()
    config.chunk_size = chunk_size
    config.discount = discount
    config.frame_stack = None
    config.p_aug = 0.0
    return config


def make_dataset(terminals=None, valids=None, chunk_size=3, discount=0.9):
    size = 8
    if terminals is None:
        terminals = np.asarray([0, 0, 0, 0, 0, 0, 0, 1], dtype=np.float32)
    if valids is None:
        valids = 1.0 - terminals
    base = Dataset.create(
        freeze=False,
        observations=np.arange(size, dtype=np.float32)[:, None],
        actions=np.arange(size, dtype=np.float32)[:, None],
        terminals=terminals,
        valids=valids,
    )
    return FixedGoalGCChunkDataset(base, make_config(chunk_size, discount))


class GCChunkDatasetTest(unittest.TestCase):
    def test_complete_chunk_has_discounted_return_and_k_step_next_state(self):
        dataset = make_dataset()
        dataset.set_goals([6], [5])

        batch = dataset.sample(1, idxs=np.asarray([1]), evaluation=True)

        np.testing.assert_array_equal(batch['actions'], np.asarray([[1.0, 2.0, 3.0]]))
        np.testing.assert_array_equal(batch['next_observations'], np.asarray([[4.0]]))
        np.testing.assert_allclose(batch['rewards'], np.asarray([-(1.0 + 0.9 + 0.9**2)]), rtol=1e-6)
        np.testing.assert_array_equal(batch['masks'], np.asarray([1.0]))

    def test_goal_inside_chunk_stops_rewards_and_bootstrap(self):
        dataset = make_dataset()
        dataset.set_goals([2], [5])

        batch = dataset.sample(1, idxs=np.asarray([1]), evaluation=True)

        np.testing.assert_allclose(batch['rewards'], np.asarray([-1.0]))
        np.testing.assert_array_equal(batch['masks'], np.asarray([0.0]))

    def test_k_one_reduces_to_atomic_gciql_transition(self):
        dataset = make_dataset(chunk_size=1)
        dataset.set_goals([6], [5])

        batch = dataset.sample(1, idxs=np.asarray([1]), evaluation=True)

        np.testing.assert_array_equal(batch['actions'], np.asarray([[1.0]]))
        np.testing.assert_array_equal(batch['next_observations'], np.asarray([[2.0]]))
        np.testing.assert_array_equal(batch['rewards'], np.asarray([-1.0]))
        np.testing.assert_array_equal(batch['masks'], np.asarray([1.0]))

    def test_k_one_keeps_all_valid_compact_transitions(self):
        terminals = np.asarray([0, 0, 1, 1, 0, 0, 1, 1], dtype=np.float32)
        valids = np.asarray([1, 1, 1, 0, 1, 1, 1, 0], dtype=np.float32)
        dataset = make_dataset(terminals=terminals, valids=valids, chunk_size=1)

        np.testing.assert_array_equal(dataset.chunk_valid_idxs, dataset.dataset.valid_idxs)

    def test_chunk_cannot_cross_trajectory_boundary(self):
        terminals = np.asarray([0, 0, 0, 1, 0, 0, 0, 1], dtype=np.float32)
        valids = 1.0 - terminals
        dataset = make_dataset(terminals=terminals, valids=valids, chunk_size=2)
        dataset.set_goals([3], [3])

        with self.assertRaisesRegex(ValueError, 'complete, valid action chunks'):
            dataset.sample(1, idxs=np.asarray([2]), evaluation=True)

    def test_chunk_may_end_at_compact_trajectory_final_observation(self):
        terminals = np.asarray([0, 0, 1, 1, 0, 0, 1, 1], dtype=np.float32)
        valids = np.asarray([1, 1, 1, 0, 1, 1, 1, 0], dtype=np.float32)
        dataset = make_dataset(terminals=terminals, valids=valids, chunk_size=2)
        dataset.set_goals([2], [2])

        batch = dataset.sample(1, idxs=np.asarray([1]), evaluation=True)

        np.testing.assert_array_equal(batch['actions'], np.asarray([[1.0, 2.0]]))
        np.testing.assert_array_equal(batch['next_observations'], np.asarray([[3.0]]))

    def test_default_agent_is_gaussian_awr_chunking(self):
        base_config = get_gciql_config()
        config = get_config()

        self.assertEqual(config.agent_name, 'gciql_chunk')
        self.assertEqual(config.dataset_class, 'GCChunkDataset')
        self.assertEqual(config.chunk_size, 5)
        self.assertEqual(config.actor_loss, 'awr')
        self.assertEqual(config.alpha, 3.0)
        self.assertFalse(config.discrete)
        self.assertEqual(set(config.keys()), set(base_config.keys()) | {'chunk_size'})
        intended_changes = {'agent_name', 'dataset_class', 'actor_loss', 'alpha'}
        for key in base_config:
            if key not in intended_changes:
                self.assertEqual(config[key], base_config[key], key)

    def test_non_chunk_specific_methods_are_inherited_from_gciql(self):
        for method_name in (
            'expectile_loss',
            'value_loss',
            'actor_loss',
            'total_loss',
            'target_update',
            'update',
            'sample_actions',
        ):
            self.assertIs(getattr(GCIQLChunkAgent, method_name), getattr(GCIQLAgent, method_name), method_name)

    def test_agent_updates_and_emits_flattened_action_chunk(self):
        config = get_config()
        config.encoder = None
        config.chunk_size = 3
        config.actor_hidden_dims = (16, 16)
        config.value_hidden_dims = (16, 16)
        observations = jnp.zeros((2, 4), dtype=jnp.float32)
        actions = jnp.zeros((2, 6), dtype=jnp.float32)
        agent = GCIQLChunkAgent.create(0, observations, actions, config)
        batch = {
            'observations': observations,
            'next_observations': jnp.ones_like(observations),
            'value_goals': jnp.ones_like(observations),
            'actor_goals': jnp.ones_like(observations),
            'actions': actions,
            'rewards': -jnp.ones((2,), dtype=jnp.float32),
            'masks': jnp.ones((2,), dtype=jnp.float32),
        }

        critic_loss, critic_info = agent.critic_loss(batch, grad_params=None)
        next_v = agent.network.select('value')(batch['next_observations'], batch['value_goals'])
        target_q = batch['rewards'] + config.discount**config.chunk_size * batch['masks'] * next_v
        q1, q2 = agent.network.select('critic')(batch['observations'], batch['value_goals'], batch['actions'])
        expected_critic_loss = ((q1 - target_q) ** 2 + (q2 - target_q) ** 2).mean()
        agent, info = agent.update(batch)
        sampled_actions = agent.sample_actions(observations[:1], observations[:1], seed=agent.rng)

        np.testing.assert_allclose(critic_loss, expected_critic_loss, rtol=1e-6)
        np.testing.assert_allclose(critic_info['q_mean'], target_q.mean(), rtol=1e-6)
        self.assertEqual(sampled_actions.shape, (1, 6))
        self.assertIn('value/value_loss', info)
        self.assertIn('critic/critic_loss', info)
        self.assertIn('actor/actor_loss', info)

    def test_k_one_losses_match_original_gciql(self):
        base_config = get_gciql_config()
        base_config.encoder = None
        base_config.actor_loss = 'awr'
        base_config.alpha = 3.0
        base_config.actor_hidden_dims = (16, 16)
        base_config.value_hidden_dims = (16, 16)
        chunk_config = get_config()
        chunk_config.encoder = None
        chunk_config.chunk_size = 1
        chunk_config.actor_hidden_dims = base_config.actor_hidden_dims
        chunk_config.value_hidden_dims = base_config.value_hidden_dims
        observations = jnp.zeros((2, 4), dtype=jnp.float32)
        actions = jnp.zeros((2, 2), dtype=jnp.float32)
        batch = {
            'observations': observations,
            'next_observations': jnp.ones_like(observations),
            'value_goals': jnp.ones_like(observations),
            'actor_goals': jnp.ones_like(observations),
            'actions': actions,
            'rewards': -jnp.ones((2,), dtype=jnp.float32),
            'masks': jnp.ones((2,), dtype=jnp.float32),
        }
        base_agent = GCIQLAgent.create(0, observations, actions, base_config)
        chunk_agent = GCIQLChunkAgent.create(0, observations, actions, chunk_config)

        base_loss, base_info = base_agent.total_loss(batch, grad_params=None)
        chunk_loss, chunk_info = chunk_agent.total_loss(batch, grad_params=None)

        np.testing.assert_allclose(base_loss, chunk_loss, rtol=1e-6)
        for key in base_info:
            np.testing.assert_allclose(base_info[key], chunk_info[key], rtol=1e-6, err_msg=key)


if __name__ == '__main__':
    unittest.main()
