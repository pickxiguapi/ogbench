import numpy as np

from precompute_lewm_npz_latents import (
    episode_layout_from_terminals,
    load_npz_source,
)


def test_episode_layout_from_terminal_rows_and_unterminated_tail():
    episode_idx, step_idx, offsets, lengths = episode_layout_from_terminals(
        np.asarray([False, True, False, False, True, False])
    )
    np.testing.assert_array_equal(offsets, [0, 2, 5])
    np.testing.assert_array_equal(lengths, [2, 3, 1])
    np.testing.assert_array_equal(episode_idx, [0, 0, 1, 1, 1, 2])
    np.testing.assert_array_equal(step_idx, [0, 1, 0, 1, 2, 0])


def test_episode_layout_accepts_final_terminal_without_empty_episode():
    episode_idx, step_idx, offsets, lengths = episode_layout_from_terminals(
        np.asarray([False, True, False, True])
    )
    np.testing.assert_array_equal(offsets, [0, 2])
    np.testing.assert_array_equal(lengths, [2, 2])
    np.testing.assert_array_equal(episode_idx, [0, 0, 1, 1])
    np.testing.assert_array_equal(step_idx, [0, 1, 0, 1])


def test_load_npz_source_validates_rows_and_builds_layout(tmp_path):
    path = tmp_path / 'visual-test-v0.npz'
    np.savez_compressed(
        path,
        observations=np.zeros((4, 64, 64, 3), dtype=np.uint8),
        actions=np.zeros((4, 5), dtype=np.float32),
        terminals=np.asarray([False, True, False, True]),
        qpos=np.zeros((4, 2), dtype=np.float32),
    )
    archive, arrays, layout = load_npz_source(path)
    try:
        assert sorted(arrays) == ['actions', 'observations', 'qpos', 'terminals']
        np.testing.assert_array_equal(layout[2], [0, 2])
        np.testing.assert_array_equal(layout[3], [2, 2])
    finally:
        archive.close()
