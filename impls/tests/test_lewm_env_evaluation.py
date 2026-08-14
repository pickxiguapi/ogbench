from types import SimpleNamespace

import h5py
import numpy as np

from eval_ogbench_agent_lewm_envs import OGBenchAgentPolicy
from ogbench.lewm_envs.evaluation import HDF5EvaluationDataset, StandardActionScaler, evaluate_dataset_goals
from utils.evaluation import evaluate


class _Dataset:
    def __init__(self):
        self.rows = {
            (0, 0): {
                'proprio': np.array([60.0, 112.0], dtype=np.float32),
                'pixels': np.full((224, 224, 3), 10, dtype=np.uint8),
            },
            (0, 1): {
                'proprio': np.array([64.0, 112.0], dtype=np.float32),
                'pixels': np.full((224, 224, 3), 20, dtype=np.uint8),
            },
        }

    def row(self, episode, step):
        return self.rows[(int(episode), int(step))]


class _ZeroPolicy:
    def reset(self, action_space, num_envs):
        self.shape = action_space.shape
        self.num_envs = num_envs

    def get_actions(self, pixels, goals, alive):
        assert pixels.shape == (self.num_envs, 1, 224, 224, 3)
        assert goals.shape == pixels.shape
        return np.zeros((self.num_envs, *self.shape), dtype=np.float32)


def test_standard_action_scaler_matches_population_statistics():
    scaler = StandardActionScaler(np.array([[1.0, 3.0], [3.0, 7.0], [np.nan, np.nan]]))
    np.testing.assert_allclose(scaler.mean, [2.0, 5.0])
    np.testing.assert_allclose(scaler.scale, [1.0, 2.0])
    value = np.array([[2.5, 9.0]])
    np.testing.assert_allclose(scaler.inverse_transform(scaler.transform(value)), value)


def test_hdf5_dataset_uses_episode_offsets_without_materializing_row_map(tmp_path):
    path = tmp_path / 'tiny.h5'
    with h5py.File(path, 'w') as file:
        file['ep_idx'] = np.array([10, 10, 10, 20, 20, 20, 20], dtype=np.int32)
        file['step_idx'] = np.array([0, 1, 2, 0, 1, 2, 3], dtype=np.int64)
        file['ep_offset'] = np.array([0, 3], dtype=np.int64)
        file['ep_len'] = np.array([3, 4], dtype=np.int32)
        file['value'] = np.arange(7, dtype=np.int32)

    dataset = HDF5EvaluationDataset(path)
    try:
        assert not hasattr(dataset, '_row_for_step')
        assert dataset.row(20, 2)['value'] == 5
        episodes, starts = dataset.sample_starts(num_eval=2, goal_offset=1, seed=7)
        np.testing.assert_array_equal(episodes, [20, 20])
        np.testing.assert_array_equal(starts, [0, 1])
        assert len(set(zip(episodes.tolist(), starts.tolist()))) == 2
    finally:
        dataset.close()


def test_tworoom_dataset_goal_evaluation_smoke():
    result = evaluate_dataset_goals(
        task='tworoom',
        dataset=_Dataset(),
        episodes=np.array([0]),
        starts=np.array([0]),
        goal_offset=1,
        eval_budget=1,
        policy=_ZeroPolicy(),
    )
    assert result['success_rate'] == 100.0
    np.testing.assert_array_equal(result['episode_successes'], [True])
    assert result['seeds'] == [None]


class _ChunkAgent:
    action_horizon = 2

    def sample_actions(self, observations, goals, seed, temperature):
        assert observations.shape == (2, 8, 8, 3)
        assert goals.shape == observations.shape
        assert temperature == 0.0
        assert seed.shape == (2,)
        return np.array([[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0]], dtype=np.float32)


def test_ogbench_agent_policy_uses_explicit_action_horizon():
    scaler = StandardActionScaler(np.array([[-1.0, -2.0], [1.0, 2.0]], dtype=np.float32))
    policy = OGBenchAgentPolicy(_ChunkAgent(), scaler, seed=0)
    policy.reset(SimpleNamespace(shape=(2,)), num_envs=2)
    pixels = np.zeros((2, 1, 8, 8, 3), dtype=np.uint8)
    goals = np.ones_like(pixels)
    first = policy.get_actions(pixels, goals, np.array([True, True]))
    second = policy.get_actions(pixels, goals, np.array([True, False]))
    np.testing.assert_allclose(first, [[0.0, 2.0], [4.0, 10.0]])
    np.testing.assert_allclose(second[0], [2.0, 6.0])
    assert np.isnan(second[1]).all()


class _AtomicAgentWithUnrelatedChunkConfig:
    def sample_actions(self, observations, goals, seed, temperature):
        return np.array([0.25, -0.5], dtype=np.float32)


class _OneStepEnv:
    def reset(self, options):
        return np.zeros(3, dtype=np.float32), {'goal': np.ones(3, dtype=np.float32)}

    def step(self, action):
        np.testing.assert_allclose(action, [0.25, -0.5])
        return np.zeros(3, dtype=np.float32), 0.0, True, False, {'success': 1.0}


def test_public_evaluation_ignores_config_chunk_size_without_agent_capability():
    stats, trajectories, renders = evaluate(
        _AtomicAgentWithUnrelatedChunkConfig(),
        _OneStepEnv(),
        config={'discrete': False, 'chunk_size': 5},
        num_eval_episodes=1,
    )
    assert stats['success'] == 1.0
    assert len(trajectories) == 1
    assert renders == []
