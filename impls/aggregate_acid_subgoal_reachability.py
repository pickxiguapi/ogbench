"""Aggregate ACID/reachability JSON files across predictor and evaluation seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


METRICS = (
    'closed_loop_success_rate',
    'acid_error_mean',
    'acid_first_block_error_mean',
    'real_min_subgoal_mse_mean',
    'relative_min_subgoal_mse_mean',
    'reach_at_0.50',
    'reach_at_0.25',
    'first_block_realization_mse_mean',
    'imagined_subgoal_mse_mean',
    'acid_vs_relative_distance_spearman',
    'acid_predicts_reach_0.50_auc',
    'acid_predicts_reach_0.25_auc',
    'acid_first_block_vs_realization_error_pearson',
    'acid_first_block_vs_realization_error_spearman',
    'acid_first_block_predicts_high_realization_error_auc',
    'first_block_realization_mse_at_25pct_acid_coverage',
    'first_block_realization_mse_at_50pct_acid_coverage',
    'first_block_realization_mse_at_75pct_acid_coverage',
    'first_block_realization_mse_at_100pct_acid_coverage',
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument(
        '--architectures', nargs='+', default=('history_mlp', 'endpoint_flow', 'latent_path_flow')
    )
    parser.add_argument('--train-seeds', type=int, nargs='+', default=(0, 1, 42))
    parser.add_argument('--eval-seeds', type=int, nargs='+', default=(0, 1, 42))
    parser.add_argument('--tasks', nargs='+', default=('tworoom', 'reacher', 'pusht', 'cube'))
    return parser.parse_args()


def main():
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    cells = {}
    missing = []
    for architecture in args.architectures:
        for train_seed in args.train_seeds:
            for eval_seed in args.eval_seeds:
                for task in args.tasks:
                    path = (
                        root
                        / architecture
                        / f'train{train_seed}'
                        / f'eval{eval_seed}'
                        / task
                        / 'reachability.json'
                    )
                    if not path.is_file():
                        missing.append(str(path))
                        continue
                    payload = json.loads(path.read_text())
                    cells[(architecture, train_seed, eval_seed, task)] = payload[
                        'metrics'
                    ]
    if missing:
        raise FileNotFoundError(
            f'Missing {len(missing)} reachability results; first: {missing[0]}'
        )

    summary = {
        'root': str(root),
        'aggregation': (
            'average evaluation seeds within each predictor-training seed, then '
            'report mean and population std over predictor-training seeds'
        ),
        'architectures': {},
    }
    for architecture in args.architectures:
        architecture_summary = {}
        for task in args.tasks:
            task_summary = {}
            for metric in METRICS:
                train_values = []
                for train_seed in args.train_seeds:
                    values = [
                        cells[(architecture, train_seed, eval_seed, task)][metric]
                        for eval_seed in args.eval_seeds
                    ]
                    values = [
                        float('nan') if value is None else float(value)
                        for value in values
                    ]
                    train_values.append(float(np.nanmean(values)))
                task_summary[metric] = {
                    'mean': float(np.nanmean(train_values)),
                    'std': float(np.nanstd(train_values)),
                    'training_seed_values': train_values,
                }
            architecture_summary[task] = task_summary
        summary['architectures'][architecture] = architecture_summary

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
