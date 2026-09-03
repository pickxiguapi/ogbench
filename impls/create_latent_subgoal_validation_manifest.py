"""Create one immutable fixed-offset latent-subgoal validation manifest."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from utils.latent_subgoal_dataset import (
    build_fixed_offset_validation_manifest,
    load_latent_cache,
    split_episodes,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--latent-dataset', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--split-seed', type=int, default=0)
    parser.add_argument('--train-fraction', type=float, default=0.95)
    parser.add_argument('--num-pairs', type=int, default=10_000)
    parser.add_argument('--history-size', type=int, default=3)
    parser.add_argument('--goal-offset', type=int, default=50)
    parser.add_argument('--subgoal-steps', type=int, default=10)
    parser.add_argument('--action-block', type=int, default=5)
    parser.add_argument('--seed', type=int, default=1)
    return parser.parse_args()


def manifest_payload(args):
    dataset_path = Path(args.latent_dataset).expanduser().resolve()
    cache = load_latent_cache(dataset_path)
    _, val_episodes = split_episodes(
        len(cache.episode_offsets), args.train_fraction, args.split_seed
    )
    arrays = build_fixed_offset_validation_manifest(
        cache.episode_offsets,
        cache.episode_lengths,
        val_episodes,
        num_pairs=args.num_pairs,
        history_size=args.history_size,
        goal_offset=args.goal_offset,
        subgoal_steps=args.subgoal_steps,
        action_block=args.action_block,
        seed=args.seed,
    )
    metadata = {
        'format': 'lewm_latent_subgoal_validation_manifest',
        'latent_dataset': str(dataset_path),
        'lewm_checkpoint_sha256': str(
            cache.metadata.get('checkpoint_sha256', '')
        ),
        'num_rows': len(cache.z),
        'num_episodes': len(cache.episode_offsets),
        'validation_episode_ids': np.asarray(val_episodes, dtype=np.int32).tolist(),
        'split_seed': args.split_seed,
        'train_fraction': args.train_fraction,
        'num_pairs': args.num_pairs,
        'history_size': args.history_size,
        'goal_offset': args.goal_offset,
        'subgoal_steps': args.subgoal_steps,
        'action_block': args.action_block,
        'sampling_seed': args.seed,
    }
    return arrays, metadata


def main():
    args = parse_args()
    arrays, metadata = manifest_payload(args)
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    if output.exists():
        with np.load(output, allow_pickle=False) as existing:
            existing_metadata = json.loads(str(existing['metadata_json'].item()))
            if existing_metadata != metadata:
                raise ValueError(f'Existing manifest metadata mismatch: {output}')
            for name, expected in arrays.items():
                if name not in existing or not np.array_equal(existing[name], expected):
                    raise ValueError(
                        f'Existing manifest array mismatch for {name}: {output}'
                    )
        print(f'Validated existing manifest: {output}', flush=True)
        return

    temporary = output.with_suffix(output.suffix + '.tmp')
    with temporary.open('wb') as file:
        np.savez_compressed(
            file,
            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
            **arrays,
        )
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary, output)
    print(f'Created validation manifest: {output}', flush=True)


if __name__ == '__main__':
    main()
