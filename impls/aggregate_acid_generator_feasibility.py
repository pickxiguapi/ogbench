"""Aggregate paired fixed-context action-feasibility results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


METRICS = (
    'acid_mean_cost_mean',
    'acid_max_cost_mean',
    'forward_mean_block_mse',
    'forward_max_block_mse',
    'relative_real_min_subgoal_mse',
    'real_subgoal_reach_at_0.50',
    'real_subgoal_reach_at_0.25',
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--architectures', nargs='+', required=True)
    parser.add_argument('--tasks', nargs='+', required=True)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    summary = {'root': str(root), 'architectures': {}}
    for architecture in args.architectures:
        task_results = {}
        for task in args.tasks:
            path = root / architecture / task / 'paired.json'
            payload = json.loads(path.read_text())
            task_results[task] = {
                metric: payload['metrics'][metric] for metric in METRICS
            }
        task_results['macro_average'] = {
            metric: float(np.mean([task_results[t][metric] for t in args.tasks]))
            for metric in METRICS
        }
        summary['architectures'][architecture] = task_results

    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
