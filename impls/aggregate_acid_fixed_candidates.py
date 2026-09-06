"""Aggregate the four-task ACID fixed-candidate calibration diagnostic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


PRIMARY_METRICS = (
    'forward_max_block_mse',
    'acid_max_vs_forward_max_spearman',
    'acid_max_vs_forward_max_within_state_spearman_mean',
    'acid_max_predicts_within_state_high_forward_error_auc',
    'forward_max_block_mse_at_25pct_acid_coverage',
    'forward_max_block_mse_at_50pct_acid_coverage',
    'forward_max_block_mse_at_75pct_acid_coverage',
    'forward_max_block_mse_at_100pct_acid_coverage',
    'high_forward_error_rate_at_25pct_acid_coverage',
    'high_forward_error_rate_at_50pct_acid_coverage',
    'high_forward_error_rate_at_75pct_acid_coverage',
    'high_forward_error_rate_at_100pct_acid_coverage',
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument(
        '--tasks', nargs='+', default=('tworoom', 'reacher', 'pusht', 'cube')
    )
    return parser.parse_args()


def finite_mean(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(values.mean()) if len(values) else None


def main():
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    results = {}
    for task in args.tasks:
        path = root / task / 'fixed_candidates.json'
        if not path.is_file():
            raise FileNotFoundError(path)
        results[task] = json.loads(path.read_text())
    summary = {
        'protocol': 'four-task mean of fixed-candidate per-task diagnostics',
        'root': str(root),
        'tasks': list(args.tasks),
        'per_task': {
            task: {metric: results[task]['metrics'][metric] for metric in PRIMARY_METRICS}
            for task in args.tasks
        },
        'four_task_mean': {
            metric: finite_mean(
                [
                    np.nan
                    if results[task]['metrics'][metric] is None
                    else results[task]['metrics'][metric]
                    for task in args.tasks
                ]
            )
            for metric in PRIMARY_METRICS
        },
    }
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
