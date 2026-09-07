#!/usr/bin/env bash
set -euo pipefail

# Build compact, protocol-equivalent LeWM evaluation datasets for H25/H50 and
# eval seeds 0/1/42. The compact HDF5 files preserve full episode metadata and
# action statistics, but materialize state/pixel rows only for the exact starts
# and goals used by this evaluation matrix. Tiny Lance tables provide the shape
# sample needed to restore the frozen GCIQL policy without copying full tables.

CLIENT_ID=node4
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export OGBENCH_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)
source "$OGBENCH_ROOT/scripts/client_env.sh"

PYTHON_BIN=${PYTHON_BIN_OVERRIDE:-/data-training/yyf/envs/ogbench/bin/python}
SOURCE_ROOT=${SOURCE_ROOT:-/data-training/yyf/datasets/latent-geometry}
OUTPUT_ROOT=${OUTPUT_ROOT:-/data-training/yyf/datasets/lewm-eval-compact-h25-h50-seeds0-1-42-v1}

if [[ -e "$OUTPUT_ROOT" ]]; then
  echo "Refusing to overwrite existing OUTPUT_ROOT: $OUTPUT_ROOT" >&2
  exit 2
fi

building_root="${OUTPUT_ROOT}.building.$$"
mkdir -p "$building_root"

PYTHONPATH="$OGBENCH_ROOT:$OGBENCH_ROOT/impls" "$PYTHON_BIN" - \
  "$SOURCE_ROOT" "$building_root" <<'PY'
import io
import json
import pathlib
import sys

import h5py
import lancedb
import numpy as np
from PIL import Image

source_root = pathlib.Path(sys.argv[1])
output_root = pathlib.Path(sys.argv[2])
seeds = (0, 1, 42)
horizons = (25, 50)
num_eval = 50

tasks = {
    'tworoom': ('tworoom.h5', ('proprio',)),
    'pusht': ('pusht_expert_train.h5', ('state',)),
    'cube': (
        'cube_single_expert.h5',
        ('qpos', 'qvel', 'privileged_block_0_pos', 'privileged_block_0_quat'),
    ),
    'reacher': ('reacher.h5', ('qpos', 'qvel')),
}


def sample_starts(lengths, num_eval, goal_offset, seed):
    valid_counts = np.maximum(lengths - goal_offset, 0)
    cumulative = np.cumsum(valid_counts)
    total_valid = int(cumulative[-1]) if len(cumulative) else 0
    rng = np.random.default_rng(seed)
    positions = np.sort(rng.choice(total_valid - 1, size=num_eval, replace=False))
    slots = np.searchsorted(cumulative, positions, side='right')
    previous = np.where(slots == 0, 0, cumulative[slots - 1])
    return slots, positions - previous


def copy_rows(source, target, indices, batch_size=32):
    for begin in range(0, len(indices), batch_size):
        batch = indices[begin : begin + batch_size]
        target[batch] = source[batch]


for task, (filename, state_columns) in tasks.items():
    source_path = source_root / filename
    output_path = output_root / filename
    with h5py.File(source_path, 'r', swmr=True, rdcc_nbytes=512 * 1024 * 1024) as source:
        episode_col = 'episode_idx' if 'episode_idx' in source else 'ep_idx'
        offsets = np.asarray(source['ep_offset'][:], dtype=np.int64)
        lengths = np.asarray(source['ep_len'][:], dtype=np.int64)
        episodes = np.asarray(source[episode_col][offsets], dtype=np.int64)
        selected = set()
        manifest = {}
        for seed in seeds:
            for horizon in horizons:
                slots, starts = sample_starts(lengths, num_eval, horizon, seed)
                rows = offsets[slots] + starts
                selected.update(int(row) for row in rows)
                selected.update(int(row + horizon) for row in rows)
                manifest[f'seed{seed}_h{horizon}'] = {
                    'episodes': episodes[slots].astype(int).tolist(),
                    'starts': starts.astype(int).tolist(),
                }
        selected = np.asarray(sorted(selected), dtype=np.int64)

        with h5py.File(output_path, 'w', libver='latest') as target:
            target.create_dataset('ep_offset', data=offsets)
            target.create_dataset('ep_len', data=lengths)
            target.attrs['compact_eval_metadata_json'] = json.dumps(
                {
                    'source': str(source_path),
                    'seeds': seeds,
                    'horizons': horizons,
                    'num_eval': num_eval,
                    'selected_rows': int(len(selected)),
                    'manifest': manifest,
                },
                sort_keys=True,
            )

            action_source = source['action']
            action_target = target.create_dataset(
                'action',
                shape=action_source.shape,
                dtype=action_source.dtype,
                chunks=action_source.chunks,
            )
            for begin in range(0, len(action_source), 100_000):
                end = min(begin + 100_000, len(action_source))
                action_target[begin:end] = action_source[begin:end]

            episode_source = source[episode_col]
            episode_target = target.create_dataset(
                episode_col,
                shape=episode_source.shape,
                dtype=episode_source.dtype,
                chunks=(4096,),
                fillvalue=-1,
            )
            episode_rows = np.unique(np.concatenate([offsets, selected]))
            copy_rows(episode_source, episode_target, episode_rows, batch_size=4096)

            step_source = source['step_idx']
            step_target = target.create_dataset(
                'step_idx',
                shape=step_source.shape,
                dtype=step_source.dtype,
                chunks=(4096,),
                fillvalue=-1,
            )
            copy_rows(step_source, step_target, selected, batch_size=4096)

            for column in (*state_columns, 'pixels'):
                source_data = source[column]
                row_shape = source_data.shape[1:]
                if column == 'pixels':
                    target_data = target.create_dataset(
                        column,
                        shape=source_data.shape,
                        dtype=source_data.dtype,
                        chunks=(1, *row_shape),
                        compression='gzip',
                        compression_opts=1,
                    )
                    copy_rows(source_data, target_data, selected, batch_size=16)
                else:
                    target_data = target.create_dataset(
                        column,
                        shape=source_data.shape,
                        dtype=source_data.dtype,
                        chunks=(1024, *row_shape),
                    )
                    copy_rows(source_data, target_data, selected, batch_size=4096)

        # Two short episodes are sufficient for GCChunkDataset.sample(1), and
        # the restored shared-LeWM policy never uses this table afterward.
        records = []
        for slot in range(2):
            count = min(8, int(lengths[slot]))
            rows = offsets[slot] + np.arange(count, dtype=np.int64)
            for row in rows:
                image = Image.fromarray(np.asarray(source['pixels'][row], dtype=np.uint8))
                buffer = io.BytesIO()
                image.save(buffer, format='JPEG', quality=95)
                records.append(
                    {
                        'episode_idx': int(episodes[slot]),
                        'action': np.asarray(source['action'][row], dtype=np.float32).tolist(),
                        'pixels': buffer.getvalue(),
                    }
                )
        database = lancedb.connect(str(output_root))
        database.create_table(pathlib.Path(filename).stem, records, mode='create')

    # Verify exact sampling and all selected source rows after reopening.
    with h5py.File(source_path, 'r', swmr=True) as source, h5py.File(output_path, 'r', swmr=True) as compact:
        assert np.array_equal(source['action'][:], compact['action'][:], equal_nan=True)
        for column in (*state_columns, 'pixels', episode_col, 'step_idx'):
            assert np.array_equal(source[column][selected], compact[column][selected])
        for seed in seeds:
            for horizon in horizons:
                slots, starts = sample_starts(lengths, num_eval, horizon, seed)
                expected_episodes = episodes[slots]
                compact_episodes = compact[episode_col][compact['ep_offset'][:]][slots]
                assert np.array_equal(expected_episodes, compact_episodes)
                assert len(starts) == num_eval
    print(f'BUILT task={task} selected_rows={len(selected)} output={output_path}', flush=True)

(output_root / 'DONE').write_text('complete\n')
PY

mv "$building_root" "$OUTPUT_ROOT"
echo "COMPACT_DATA_DONE output=$OUTPUT_ROOT"
