"""Validate that a Visual OGBench generator matches its frozen LeWM."""

import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--generator', required=True)
    parser.add_argument('--lewm', required=True)
    args = parser.parse_args()

    generator = Path(args.generator).resolve()
    lewm = Path(args.lewm).resolve()
    config_path = generator.parent / 'config.json'
    config = json.loads(config_path.read_text())
    expected = {
        'architecture': 'latent_path_flow_transformer_encoder',
        'goal_sampling': 'hiql_uniform_future_same_trajectory',
        'max_goal_steps': None,
        'subgoal_steps': 10,
        'action_block': 5,
        'history_size': 3,
        'flow_sampling_steps': 16,
        'flow_solver': 'euler',
        'train_steps': 200000,
        'num_samples': 1,
        'lewm_checkpoint_sha256': hashlib.sha256(lewm.read_bytes()).hexdigest(),
    }
    mismatches = {
        key: {'actual': config.get(key), 'expected': value}
        for key, value in expected.items()
        if config.get(key) != value
    }
    if mismatches:
        raise ValueError(f'Generator config mismatch in {config_path}: {mismatches}')


if __name__ == '__main__':
    main()
