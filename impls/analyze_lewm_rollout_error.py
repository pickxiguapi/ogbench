"""Measure frozen LeWM open-loop latent rollout error on offline trajectories."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import h5py
import jax
import jax.numpy as jnp
import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

from lewm_jax import load_frozen_lewm


TASK_COLORS = {
    'cube': '#4C78A8',
    'pusht': '#F58518',
    'reacher': '#54A24B',
    'tworoom': '#E45756',
}
TASK_LABELS = {
    'cube': 'Cube',
    'pusht': 'PushT',
    'reacher': 'Reacher',
    'tworoom': 'TwoRoom',
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tasks', nargs='+', required=True)
    parser.add_argument('--latent-datasets', nargs='+', required=True)
    parser.add_argument('--checkpoints', nargs='+', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--max-horizon', type=int, default=50)
    parser.add_argument('--action-block', type=int, default=5)
    parser.add_argument('--local-horizon', type=int, default=10)
    parser.add_argument('--num-trajectories', type=int, default=512)
    parser.add_argument('--episode-holdout-fraction', type=float, default=0.1)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--bootstrap-samples', type=int, default=1000)
    return parser.parse_args()


def sha256_file(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open('rb') as file:
        while chunk := file.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def streaming_action_stats(dataset, chunk_rows=262_144):
    """Match LeWM training's finite-row sample mean/std without a full copy."""
    count = 0
    total = None
    total_square = None
    for start in range(0, len(dataset), chunk_rows):
        values = np.asarray(dataset[start : start + chunk_rows])
        values = values[~np.isnan(values).any(axis=1)].astype(np.float64, copy=False)
        if not len(values):
            continue
        if total is None:
            total = np.zeros(values.shape[1], dtype=np.float64)
            total_square = np.zeros(values.shape[1], dtype=np.float64)
        count += len(values)
        total += values.sum(axis=0)
        total_square += np.square(values).sum(axis=0)
    if count < 2:
        raise ValueError('Need at least two finite action rows for normalization.')
    mean = total / count
    variance = (total_square - count * np.square(mean)) / (count - 1)
    std = np.sqrt(np.maximum(variance, 0.0))
    std = np.where(std > 0, std, 1.0)
    return mean, std, count


def choose_episode_starts(
    episode_offsets,
    episode_lengths,
    episode_ids,
    *,
    max_horizon,
    num_trajectories,
    holdout_fraction,
    seed,
):
    """Choose one start per trajectory from a deterministic episode subset."""
    if not 0.0 < holdout_fraction <= 1.0:
        raise ValueError('episode_holdout_fraction must be in (0, 1].')
    eligible = np.flatnonzero(np.asarray(episode_lengths) > max_horizon)
    if not len(eligible):
        raise ValueError(f'No trajectory is longer than max_horizon={max_horizon}.')
    rng = np.random.default_rng(seed)
    heldout_size = max(1, math.ceil(holdout_fraction * len(eligible)))
    heldout_slots = rng.permutation(eligible)[:heldout_size]
    if num_trajectories > len(heldout_slots):
        raise ValueError(
            f'Requested {num_trajectories} trajectories, but the deterministic '
            f'episode subset contains only {len(heldout_slots)}.'
        )
    selected_slots = heldout_slots[:num_trajectories]
    relative_starts = np.asarray(
        [
            rng.integers(0, int(episode_lengths[slot]) - max_horizon)
            for slot in selected_slots
        ],
        dtype=np.int64,
    )
    absolute_starts = np.asarray(episode_offsets)[selected_slots] + relative_starts
    selected_ids = np.asarray(episode_ids)[selected_slots]
    order = np.argsort(absolute_starts)
    return {
        'episode_slots': selected_slots[order],
        'episode_ids': selected_ids[order],
        'relative_starts': relative_starts[order],
        'absolute_starts': absolute_starts[order],
        'eligible_episodes': int(len(eligible)),
        'heldout_episodes': int(len(heldout_slots)),
    }


def take_rows(dataset, indices):
    """Read arbitrary HDF5 rows while satisfying h5py's sorted-index rule."""
    indices = np.asarray(indices, dtype=np.int64)
    unique, inverse = np.unique(indices.reshape(-1), return_inverse=True)
    values = np.asarray(dataset[unique])
    return values[inverse].reshape(*indices.shape, *values.shape[1:])


def make_rollout_function(model, variables, history_size):
    """Mirror planner autoregression, starting from the single observed z_t."""

    @jax.jit
    def rollout(initial_embeddings, action_blocks):
        embeddings = initial_embeddings[:, None]
        actions = action_blocks[:, :1]
        predictions = []
        for step in range(action_blocks.shape[1]):
            prediction = model.apply(
                variables,
                embeddings[:, -history_size:],
                actions[:, -history_size:],
                train=False,
                method=model.predict_embeddings,
            )[:, -1].astype(jnp.float32)
            predictions.append(prediction)
            embeddings = jnp.concatenate([embeddings, prediction[:, None]], axis=1)
            if step + 1 < action_blocks.shape[1]:
                actions = jnp.concatenate(
                    [actions, action_blocks[:, step + 1 : step + 2]], axis=1
                )
        return jnp.stack(predictions, axis=1)

    return rollout


def bootstrap_summary(values, persistence, *, samples, seed):
    """Return mean, CI, persistence-relative, and first-step-relative summaries."""
    values = np.asarray(values, dtype=np.float64)
    persistence = np.asarray(persistence, dtype=np.float64)
    if values.shape != persistence.shape or values.ndim != 2:
        raise ValueError('values and persistence must have matching [N, H] shapes.')
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(samples, len(values)))
    value_means = values[indices].mean(axis=1)
    persistence_means = persistence[indices].mean(axis=1)
    ratios = value_means / np.maximum(persistence_means, 1e-12)
    amplification = value_means / np.maximum(value_means[:, :1], 1e-12)
    mean = values.mean(axis=0)
    return {
        'mean': mean,
        'se': values.std(axis=0, ddof=1) / np.sqrt(len(values)),
        'ci_low': np.percentile(value_means, 2.5, axis=0),
        'ci_high': np.percentile(value_means, 97.5, axis=0),
        'relative_mean': values.mean(axis=0)
        / np.maximum(persistence.mean(axis=0), 1e-12),
        'relative_ci_low': np.percentile(ratios, 2.5, axis=0),
        'relative_ci_high': np.percentile(ratios, 97.5, axis=0),
        'persistence_mean': persistence.mean(axis=0),
        'amplification_mean': mean / max(mean[0], 1e-12),
        'amplification_ci_low': np.percentile(amplification, 2.5, axis=0),
        'amplification_ci_high': np.percentile(amplification, 97.5, axis=0),
    }


def evaluate_task(
    task,
    latent_dataset,
    checkpoint,
    *,
    max_horizon,
    action_block,
    num_trajectories,
    holdout_fraction,
    batch_size,
    seed,
    bootstrap_samples,
):
    if max_horizon <= 0 or action_block <= 0 or max_horizon % action_block:
        raise ValueError('max_horizon must be positive and divisible by action_block.')
    checkpoint_hash = sha256_file(checkpoint)
    model, variables, checkpoint_metadata = load_frozen_lewm(checkpoint)
    config = checkpoint_metadata['config']
    if int(config['frameskip']) != action_block:
        raise ValueError(
            f'{task}: checkpoint frameskip={config["frameskip"]} but '
            f'action_block={action_block}.'
        )
    num_blocks = max_horizon // action_block
    horizons = np.arange(1, num_blocks + 1, dtype=np.int64) * action_block
    rollout = make_rollout_function(model, variables, int(config['history_size']))

    with h5py.File(latent_dataset, 'r') as file:
        if file.attrs.get('status') != 'complete':
            raise ValueError(f'{task}: latent dataset is not marked complete.')
        cache_hash = str(file.attrs['checkpoint_sha256'])
        if checkpoint_hash != cache_hash:
            raise ValueError(
                f'{task}: checkpoint/cache SHA mismatch: {checkpoint_hash} != {cache_hash}'
            )
        offsets = np.asarray(file['ep_offset'][:], dtype=np.int64)
        lengths = np.asarray(file['ep_len'][:], dtype=np.int64)
        episode_column = 'episode_idx' if 'episode_idx' in file else 'ep_idx'
        episode_ids = take_rows(file[episode_column], offsets)
        selection = choose_episode_starts(
            offsets,
            lengths,
            episode_ids,
            max_horizon=max_horizon,
            num_trajectories=num_trajectories,
            holdout_fraction=holdout_fraction,
            seed=seed,
        )
        starts = selection['absolute_starts']
        initial = take_rows(file['z'], starts).astype(np.float32)
        target_rows = starts[:, None] + horizons[None]
        targets = take_rows(file['z'], target_rows).astype(np.float32)
        action_mean, action_std, action_stat_rows = streaming_action_stats(file['action'])
        action_rows = (
            starts[:, None, None]
            + np.arange(num_blocks, dtype=np.int64)[None, :, None] * action_block
            + np.arange(action_block, dtype=np.int64)[None, None, :]
        )
        actions = take_rows(file['action'], action_rows)
        actions = (actions - action_mean) / action_std
        actions = np.nan_to_num(actions, nan=0.0, posinf=0.0, neginf=0.0)
        actions = actions.reshape(len(starts), num_blocks, -1).astype(np.float32)

        predictions = []
        for start in range(0, len(initial), batch_size):
            stop = min(start + batch_size, len(initial))
            valid = stop - start
            batch_initial = initial[start:stop]
            batch_actions = actions[start:stop]
            if valid < batch_size:
                batch_initial = np.pad(batch_initial, ((0, batch_size - valid), (0, 0)))
                batch_actions = np.pad(
                    batch_actions, ((0, batch_size - valid), (0, 0), (0, 0))
                )
            prediction = np.asarray(rollout(batch_initial, batch_actions))[:valid]
            predictions.append(prediction)
        predictions = np.concatenate(predictions, axis=0)

        latent_mse = np.mean(np.square(predictions - targets), axis=-1)
        persistence_mse = np.mean(
            np.square(initial[:, None] - targets), axis=-1
        )
        numerator = np.sum(predictions * targets, axis=-1)
        denominator = np.linalg.norm(predictions, axis=-1) * np.linalg.norm(
            targets, axis=-1
        )
        cosine_error = 1.0 - numerator / np.maximum(denominator, 1e-12)
        summary = bootstrap_summary(
            latent_mse,
            persistence_mse,
            samples=bootstrap_samples,
            seed=seed + 10_000,
        )
        cosine_summary = bootstrap_summary(
            cosine_error,
            np.ones_like(cosine_error),
            samples=bootstrap_samples,
            seed=seed + 20_000,
        )
        return {
            'task': task,
            'horizons': horizons,
            'selection': selection,
            'latent_mse': latent_mse,
            'persistence_mse': persistence_mse,
            'cosine_error': cosine_error,
            'summary': summary,
            'cosine_summary': cosine_summary,
            'metadata': {
                'latent_dataset': str(Path(latent_dataset).resolve()),
                'checkpoint': str(Path(checkpoint).resolve()),
                'checkpoint_sha256': checkpoint_hash,
                'cache_checkpoint_sha256': cache_hash,
                'checkpoint_epoch': int(checkpoint_metadata['epoch']),
                'checkpoint_seed': int(config['seed']),
                'frameskip': int(config['frameskip']),
                'history_size': int(config['history_size']),
                'action_dim': int(file['action'].shape[1]),
                'action_stat_rows': int(action_stat_rows),
                'eligible_episodes': selection['eligible_episodes'],
                'diagnostic_holdout_episodes': selection['heldout_episodes'],
                'evaluated_trajectories': int(len(starts)),
            },
        }


def write_outputs(results, output_dir, *, local_horizon, args):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / 'raw_rollout_errors.csv'
    with raw_path.open('w', newline='') as file:
        fieldnames = [
            'task',
            'episode_id',
            'start_step',
            'horizon',
            'latent_mse',
            'persistence_mse',
            'cosine_error',
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            for sample in range(len(result['selection']['episode_ids'])):
                for index, horizon in enumerate(result['horizons']):
                    writer.writerow(
                        {
                            'task': result['task'],
                            'episode_id': int(result['selection']['episode_ids'][sample]),
                            'start_step': int(result['selection']['relative_starts'][sample]),
                            'horizon': int(horizon),
                            'latent_mse': float(result['latent_mse'][sample, index]),
                            'persistence_mse': float(
                                result['persistence_mse'][sample, index]
                            ),
                            'cosine_error': float(result['cosine_error'][sample, index]),
                        }
                    )

    summary_path = output_dir / 'summary.csv'
    with summary_path.open('w', newline='') as file:
        fieldnames = [
            'task',
            'horizon',
            'n',
            'latent_mse_mean',
            'latent_mse_se',
            'latent_mse_ci_low',
            'latent_mse_ci_high',
            'persistence_mse_mean',
            'relative_mse',
            'relative_mse_ci_low',
            'relative_mse_ci_high',
            'error_amplification_vs_5step',
            'error_amplification_ci_low',
            'error_amplification_ci_high',
            'cosine_error_mean',
            'cosine_error_se',
            'cosine_error_ci_low',
            'cosine_error_ci_high',
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            summary = result['summary']
            cosine = result['cosine_summary']
            for index, horizon in enumerate(result['horizons']):
                writer.writerow(
                    {
                        'task': result['task'],
                        'horizon': int(horizon),
                        'n': len(result['latent_mse']),
                        'latent_mse_mean': float(summary['mean'][index]),
                        'latent_mse_se': float(summary['se'][index]),
                        'latent_mse_ci_low': float(summary['ci_low'][index]),
                        'latent_mse_ci_high': float(summary['ci_high'][index]),
                        'persistence_mse_mean': float(
                            summary['persistence_mean'][index]
                        ),
                        'relative_mse': float(summary['relative_mean'][index]),
                        'relative_mse_ci_low': float(
                            summary['relative_ci_low'][index]
                        ),
                        'relative_mse_ci_high': float(
                            summary['relative_ci_high'][index]
                        ),
                        'error_amplification_vs_5step': float(
                            summary['amplification_mean'][index]
                        ),
                        'error_amplification_ci_low': float(
                            summary['amplification_ci_low'][index]
                        ),
                        'error_amplification_ci_high': float(
                            summary['amplification_ci_high'][index]
                        ),
                        'cosine_error_mean': float(cosine['mean'][index]),
                        'cosine_error_se': float(cosine['se'][index]),
                        'cosine_error_ci_low': float(cosine['ci_low'][index]),
                        'cosine_error_ci_high': float(cosine['ci_high'][index]),
                    }
                )

    metadata = {
        'protocol': {
            'metric': 'open-loop autoregressive latent MSE',
            'primary_plot_normalization': 'mean latent MSE divided by mean 5-step latent MSE',
            'supplementary_normalization': 'ratio of mean LeWM MSE to mean z_t persistence MSE',
            'planner_match': 'single observed z_t followed by frozen predictor autoregression',
            'local_horizon': local_horizon,
            'max_horizon': args.max_horizon,
            'action_block': args.action_block,
            'num_trajectories_per_task': args.num_trajectories,
            'episode_holdout_fraction': args.episode_holdout_fraction,
            'seed': args.seed,
            'bootstrap_samples': args.bootstrap_samples,
            'holdout_limitation': (
                'The frozen checkpoints were trained with a clip-level 90/10 split, '
                'so this deterministic episode subset is held out from diagnostic '
                'sampling, not guaranteed unseen during LeWM training.'
            ),
        },
        'tasks': {result['task']: result['metadata'] for result in results},
    }
    with (output_dir / 'metadata.json').open('w') as file:
        json.dump(metadata, file, indent=2)
        file.write('\n')

    plot_results(results, output_dir, local_horizon=local_horizon)
    return raw_path, summary_path


def plot_results(results, output_dir, *, local_horizon):
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
    figure, axis = plt.subplots(figsize=(6.4, 4.0), constrained_layout=True)
    amplification_curves = []
    for result in results:
        task = result['task']
        horizons = result['horizons']
        summary = result['summary']
        color = TASK_COLORS.get(task)
        axis.plot(
            horizons,
            summary['amplification_mean'],
            marker='o',
            markersize=3.5,
            linewidth=1.6,
            color=color,
            label=TASK_LABELS.get(task, task),
        )
        axis.fill_between(
            horizons,
            summary['amplification_ci_low'],
            summary['amplification_ci_high'],
            color=color,
            alpha=0.12,
            linewidth=0,
        )
        amplification_curves.append(summary['amplification_mean'])
    macro = np.mean(np.stack(amplification_curves), axis=0)
    axis.plot(
        results[0]['horizons'],
        macro,
        color='#222222',
        linewidth=2.5,
        label='Task mean',
        zorder=10,
    )
    axis.axhline(1.0, color='#777777', linestyle='--', linewidth=1.0, label='5-step error')
    axis.axvspan(0, local_horizon, color='#7A3E9D', alpha=0.055, linewidth=0)
    axis.axvline(local_horizon, color='#7A3E9D', linestyle='--', linewidth=1.5)
    axis.text(
        local_horizon + 0.6,
        0.98,
        f'LeWM++ local horizon $k={local_horizon}$',
        transform=axis.get_xaxis_transform(),
        color='#7A3E9D',
        rotation=90,
        va='top',
        ha='left',
    )
    axis.set_xlabel('Open-loop horizon (environment steps)')
    axis.set_ylabel(r'Rollout error amplification ($\times$ 5-step MSE)')
    axis.set_title('Frozen LeWM error compounds under open-loop rollout')
    axis.grid(axis='both', color='#DDDDDD', linewidth=0.6, alpha=0.8)
    axis.spines[['top', 'right']].set_visible(False)
    axis.set_xlim(int(results[0]['horizons'][0]) - 1, int(results[0]['horizons'][-1]) + 1)
    axis.set_ylim(bottom=0)
    axis.legend(ncol=2, frameon=False, loc='upper left')
    output_dir = Path(output_dir)
    figure.savefig(output_dir / 'lewm_rollout_error.png', dpi=240)
    figure.savefig(output_dir / 'lewm_rollout_error.pdf', bbox_inches='tight')
    plt.close(figure)


def main():
    args = parse_args()
    if not (
        len(args.tasks) == len(args.latent_datasets) == len(args.checkpoints)
    ):
        raise ValueError('tasks, latent-datasets, and checkpoints must have equal lengths.')
    if args.local_horizon <= 0 or args.local_horizon > args.max_horizon:
        raise ValueError('local_horizon must be in [1, max_horizon].')
    results = []
    for index, (task, dataset, checkpoint) in enumerate(
        zip(args.tasks, args.latent_datasets, args.checkpoints)
    ):
        print(f'[{task}] evaluating {args.num_trajectories} trajectories')
        result = evaluate_task(
            task,
            dataset,
            checkpoint,
            max_horizon=args.max_horizon,
            action_block=args.action_block,
            num_trajectories=args.num_trajectories,
            holdout_fraction=args.episode_holdout_fraction,
            batch_size=args.batch_size,
            seed=args.seed + index,
            bootstrap_samples=args.bootstrap_samples,
        )
        results.append(result)
        print(
            f'[{task}] error amplification at k={args.local_horizon}: '
            f'{result["summary"]["amplification_mean"][args.local_horizon // args.action_block - 1]:.2f}x; '
            f'at H={args.max_horizon}: {result["summary"]["amplification_mean"][-1]:.2f}x'
        )
    raw_path, summary_path = write_outputs(
        results, args.output_dir, local_horizon=args.local_horizon, args=args
    )
    print(f'raw={raw_path}')
    print(f'summary={summary_path}')
    print(f'figure={Path(args.output_dir) / "lewm_rollout_error.png"}')


if __name__ == '__main__':
    main()
