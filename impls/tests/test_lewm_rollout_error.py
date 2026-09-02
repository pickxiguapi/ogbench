import numpy as np

from analyze_lewm_rollout_error import (
    bootstrap_summary,
    choose_episode_starts,
    streaming_action_stats,
)


def test_episode_selection_uses_distinct_eligible_trajectories():
    offsets = np.asarray([0, 60, 130, 210, 300, 400])
    lengths = np.asarray([60, 70, 80, 90, 100, 110])
    episode_ids = np.arange(10, 16)
    selected = choose_episode_starts(
        offsets,
        lengths,
        episode_ids,
        max_horizon=50,
        num_trajectories=3,
        holdout_fraction=0.5,
        seed=42,
    )
    assert len(np.unique(selected['episode_ids'])) == 3
    assert np.all(selected['relative_starts'] >= 0)
    slots = selected['episode_slots']
    assert np.all(selected['relative_starts'] + 50 < lengths[slots])


def test_streaming_action_stats_match_numpy_sample_statistics():
    actions = np.asarray(
        [[1.0, 2.0], [3.0, 6.0], [np.nan, 4.0], [5.0, 10.0]],
        dtype=np.float64,
    )
    mean, std, count = streaming_action_stats(actions, chunk_rows=2)
    expected = actions[[0, 1, 3]]
    np.testing.assert_allclose(mean, expected.mean(axis=0))
    np.testing.assert_allclose(std, expected.std(axis=0, ddof=1))
    assert count == 3


def test_relative_error_is_ratio_of_means():
    values = np.asarray([[1.0, 4.0], [3.0, 8.0]])
    persistence = np.asarray([[2.0, 2.0], [6.0, 6.0]])
    summary = bootstrap_summary(values, persistence, samples=100, seed=0)
    np.testing.assert_allclose(summary['relative_mean'], [0.5, 1.5])
