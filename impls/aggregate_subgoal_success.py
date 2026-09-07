"""Aggregate subgoal-generator success results over evaluation seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--prefix', required=True)
    parser.add_argument('--architectures', nargs='+', required=True)
    parser.add_argument('--train-seed', type=int, required=True)
    parser.add_argument('--eval-seeds', type=int, nargs='+', required=True)
    parser.add_argument('--tasks', nargs='+', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    summary = {'root': str(root), 'architectures': {}}
    for architecture in args.architectures:
        task_values = {}
        for task in args.tasks:
            values = []
            for eval_seed in args.eval_seeds:
                pattern = (
                    f'{args.prefix}_{architecture}_train{args.train_seed}_'
                    f'eval{eval_seed}_*'
                )
                matches = list(root.glob(f'{pattern}/{task}/result.json'))
                if len(matches) != 1:
                    raise FileNotFoundError(
                        f'Expected one result for {pattern}/{task}, got {matches}'
                    )
                values.append(float(json.loads(matches[0].read_text())['success_rate']))
            task_values[task] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'eval_seed_values': values,
            }
        macro_values = [
            float(np.mean([
                task_values[task]['eval_seed_values'][index]
                for task in args.tasks
            ]))
            for index in range(len(args.eval_seeds))
        ]
        task_values['average'] = {
            'mean': float(np.mean(macro_values)),
            'std': float(np.std(macro_values)),
            'eval_seed_values': macro_values,
        }
        summary['architectures'][architecture] = task_values

    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
