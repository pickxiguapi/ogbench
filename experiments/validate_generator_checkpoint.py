"""Fail-fast metadata validation invoked by generator-based experiment Bash files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--task', required=True)
    parser.add_argument('--family', required=True)
    parser.add_argument('--generator-type', required=True)
    parser.add_argument('--goal-sampling', required=True)
    parser.add_argument('--max-goal-steps', type=int)
    return parser.parse_args()


def main():
    args = parse_args()
    protocols = {
        'goalmax25': (
            'uniform_distance_first_aligned_future_same_trajectory_stride_5_max_25',
            25,
        ),
        'general_uniform_future': ('hiql_uniform_future_same_trajectory', None),
    }
    if args.family not in protocols:
        raise ValueError(f'Unknown generator family: {args.family}')
    expected_sampling, expected_max_goal_steps = protocols[args.family]
    if (args.goal_sampling, args.max_goal_steps) != (expected_sampling, expected_max_goal_steps):
        raise ValueError(f'Invalid protocol requested for family {args.family}.')
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    config_path = checkpoint.parent / 'config.json'
    if not checkpoint.is_file():
        raise FileNotFoundError(f'Generator checkpoint not found: {checkpoint}')
    if not config_path.is_file():
        raise FileNotFoundError(f'Checkpoint-adjacent config not found: {config_path}')
    config = json.loads(config_path.read_text())
    expected = {
        'generator_family': args.family,
        'generator_type': args.generator_type,
        'goal_sampling': args.goal_sampling,
        'max_goal_steps': args.max_goal_steps,
    }
    mismatches = {
        key: {'actual': config.get(key), 'expected': value}
        for key, value in expected.items()
        if config.get(key) != value
    }
    config_task = config.get('task')
    latent_dataset = Path(config.get('latent_dataset') or '')
    if config_task not in (None, args.task):
        mismatches['task'] = {'actual': config_task, 'expected': args.task}
    if config_task is None and args.task not in latent_dataset.stem.lower():
        mismatches['latent_dataset'] = {'actual': str(latent_dataset), 'expected_task': args.task}
    if mismatches:
        raise ValueError(f'Generator config mismatch in {config_path}: {mismatches}')
    print(f'Validated generator config: task={args.task} family={args.family} type={args.generator_type}')


if __name__ == '__main__':
    main()
