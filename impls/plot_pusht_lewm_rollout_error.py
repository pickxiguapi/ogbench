"""Plot the PushT frozen-LeWM rollout diagnostic in environment steps."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary-csv', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--local-horizon', type=int, default=10)
    parser.add_argument('--action-block', type=int, default=5)
    return parser.parse_args()


def load_pusht_rows(path):
    with Path(path).open(newline='') as file:
        rows = [row for row in csv.DictReader(file) if row['task'] == 'pusht']
    if not rows:
        raise ValueError(f'No PushT rows found in {path}.')
    rows.sort(key=lambda row: int(row['horizon']))
    return rows


def main():
    args = parse_args()
    if args.local_horizon <= 0 or args.action_block <= 0:
        raise ValueError('local_horizon and action_block must be positive.')
    if args.local_horizon % args.action_block:
        raise ValueError('local_horizon must be divisible by action_block.')

    rows = load_pusht_rows(args.summary_csv)
    steps = np.asarray([int(row['horizon']) for row in rows])
    mse = np.asarray([float(row['latent_mse_mean']) for row in rows])
    ci_low = np.asarray([float(row['latent_mse_ci_low']) for row in rows])
    ci_high = np.asarray([float(row['latent_mse_ci_high']) for row in rows])
    sample_count = int(rows[0]['n'])
    if args.local_horizon not in steps:
        raise ValueError(f'local_horizon={args.local_horizon} is absent from the summary.')
    local_index = int(np.flatnonzero(steps == args.local_horizon)[0])
    local_chunks = args.local_horizon // args.action_block

    plt.rcParams.update(
        {
            'font.size': 10,
            'axes.labelsize': 11,
            'axes.titlesize': 12,
            'legend.fontsize': 9,
            'pdf.fonttype': 42,
            'ps.fonttype': 42,
        }
    )
    figure, axis = plt.subplots(figsize=(6.2, 3.9), constrained_layout=True)
    axis.axvspan(
        steps[0] - 1,
        args.local_horizon,
        color='#7A3E9D',
        alpha=0.055,
        linewidth=0,
    )
    axis.fill_between(steps, ci_low, ci_high, color='#F58518', alpha=0.18, linewidth=0)
    axis.plot(
        steps,
        mse,
        color='#F58518',
        marker='o',
        markersize=4.5,
        linewidth=2.2,
        label=f'PushT, {sample_count} trajectories',
    )
    axis.axvline(
        args.local_horizon,
        color='#7A3E9D',
        linestyle='--',
        linewidth=1.6,
    )
    axis.annotate(
        f'LeWM++ local horizon\n$k={args.local_horizon}$ env steps = {local_chunks} chunks',
        xy=(args.local_horizon, mse[local_index]),
        xytext=(args.local_horizon + 5, mse.max() * 0.46),
        color='#7A3E9D',
        arrowprops={'arrowstyle': '->', 'color': '#7A3E9D', 'linewidth': 1.2},
        ha='left',
        va='center',
    )
    axis.text(
        0.98,
        0.04,
        f'1 LeWM transition = 1 action chunk = {args.action_block} env steps',
        transform=axis.transAxes,
        color='#666666',
        ha='right',
        va='bottom',
        fontsize=8.5,
    )
    axis.set_xlabel('Open-loop rollout horizon (environment steps)')
    axis.set_ylabel('Latent prediction MSE')
    axis.set_title('PushT: frozen LeWM rollout error grows with horizon')
    if steps[-1] > 50:
        axis.set_xticks(np.arange(0, steps[-1] + 1, 10))
        axis.set_xlim(0, steps[-1] + 1)
    else:
        axis.set_xticks(steps)
        axis.set_xlim(steps[0] - 1, steps[-1] + 1)
    axis.set_ylim(bottom=0)
    axis.grid(color='#DDDDDD', linewidth=0.6, alpha=0.85)
    axis.spines[['top', 'right']].set_visible(False)
    axis.legend(frameon=False, loc='upper left')

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / 'pusht_lewm_rollout_error_env_steps.png'
    pdf_path = output_dir / 'pusht_lewm_rollout_error_env_steps.pdf'
    figure.savefig(png_path, dpi=240)
    figure.savefig(pdf_path, bbox_inches='tight')
    plt.close(figure)
    print(f'png={png_path}')
    print(f'pdf={pdf_path}')


if __name__ == '__main__':
    main()
