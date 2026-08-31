import numpy as np

from utils.latent_subgoal_dataset import (
    build_history_indices,
    build_valid_transitions,
    sample_future_pairs,
    split_episodes,
    validate_trajectory_layout,
)


def test_layout_split_and_valid_transitions_are_episode_safe():
    offsets = np.asarray([0, 4, 7])
    lengths = np.asarray([4, 3, 5])
    validate_trajectory_layout(12, offsets, lengths)
    train_episodes, val_episodes = split_episodes(3, train_fraction=2 / 3, seed=7)
    assert set(train_episodes).isdisjoint(set(val_episodes))
    assert set(np.concatenate((train_episodes, val_episodes))) == {0, 1, 2}

    current, final = build_valid_transitions(offsets, lengths, [0, 2])
    np.testing.assert_array_equal(current, [0, 1, 2, 7, 8, 9, 10])
    np.testing.assert_array_equal(final, [3, 3, 3, 11, 11, 11, 11])


def test_hiql_future_sampling_and_k10_targets():
    valid_t = np.asarray([0, 10, 20], dtype=np.int32)
    final_t = np.asarray([30, 30, 30], dtype=np.int32)
    t, g, target = sample_future_pairs(valid_t, final_t, 10000, subgoal_steps=10, seed=3)
    assert np.all(g > t)
    assert np.all(g <= 30)
    np.testing.assert_array_equal(target, np.minimum(t + 10, g))
    assert np.any(g - t < 10)
    assert np.any(g - t > 10)
    np.testing.assert_array_equal(target[g - t <= 10], g[g - t <= 10])


def test_three_frame_histories_repeat_pad_without_crossing_episodes():
    offsets = np.asarray([0, 4, 7])
    current = np.asarray([0, 1, 2, 3, 7, 8, 10], dtype=np.int32)
    history = build_history_indices(current, offsets, history_size=3)
    np.testing.assert_array_equal(
        history,
        [
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 2],
            [1, 2, 3],
            [7, 7, 7],
            [7, 7, 8],
            [8, 9, 10],
        ],
    )


def test_future_goal_sampling_is_uniform_including_endpoints():
    valid_t = np.asarray([0], dtype=np.int32)
    final_t = np.asarray([4], dtype=np.int32)
    _, goals, _ = sample_future_pairs(
        valid_t, final_t, 100_000, subgoal_steps=10, seed=11
    )
    counts = np.bincount(goals, minlength=5)[1:]
    assert np.max(np.abs(counts - counts.mean())) < 0.02 * counts.mean()
