"""Diagnose LeWM representation geometry and recalibrate BatchNorm statistics."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import flax
import jax
import jax.numpy as jnp
import numpy as np

from lewm_jax.checkpoints import load_frozen_lewm
from lewm_jax.loss import sigreg_loss
from utils.lewm_npz_sequence_dataset import LeWMNPZSequenceDataset


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--dataset-path', required=True)
    parser.add_argument('--validation-dataset-path', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--sample-batches', type=int, default=8)
    parser.add_argument('--calibration-batches', type=int, default=200)
    parser.add_argument('--seed', type=int, default=3072)
    return parser.parse_args()


def _sample_batches(dataset, indices, *, batch_size, num_batches, rng):
    sample_count = batch_size * num_batches
    if sample_count > len(indices):
        raise ValueError(
            f'Requested {sample_count} samples, but the split has only {len(indices)}.'
        )
    selected = rng.choice(indices, size=sample_count, replace=False)
    return [
        dataset.get_batch(selected[offset : offset + batch_size])
        for offset in range(0, sample_count, batch_size)
    ]


def _geometry_metrics(embedding_batches, sigreg_step, sigreg_key, prediction_losses):
    embeddings = np.concatenate(embedding_batches, axis=0).astype(np.float64)
    flat = embeddings.reshape(-1, embeddings.shape[-1])
    feature_mean = flat.mean(axis=0)
    centered = flat - feature_mean
    feature_std = centered.std(axis=0, ddof=1)
    covariance = centered.T @ centered / max(len(centered) - 1, 1)
    eigenvalues = np.maximum(np.linalg.eigvalsh(covariance), 0.0)
    eigenvalues = eigenvalues[::-1]
    eigenvalue_sum = eigenvalues.sum()
    if eigenvalue_sum > 0:
        probabilities = eigenvalues / eigenvalue_sum
        nonzero = probabilities > 0
        effective_rank = float(
            np.exp(-np.sum(probabilities[nonzero] * np.log(probabilities[nonzero])))
        )
        participation_ratio = float(
            eigenvalue_sum**2 / np.maximum(np.square(eigenvalues).sum(), 1e-30)
        )
    else:
        effective_rank = 0.0
        participation_ratio = 0.0

    pair_rng = np.random.default_rng(0)
    num_pairs = min(4096, len(flat))
    left = pair_rng.integers(0, len(flat), size=num_pairs)
    right = pair_rng.integers(0, len(flat), size=num_pairs)
    right = np.where(right == left, (right + 1) % len(flat), right)
    distances = np.linalg.norm(flat[left] - flat[right], axis=-1)

    sigreg_values = []
    for batch_index, batch_embeddings in enumerate(embedding_batches):
        key = jax.random.fold_in(sigreg_key, batch_index)
        value = sigreg_step(jnp.asarray(batch_embeddings), key)
        sigreg_values.append(float(jax.device_get(value)))

    largest = eigenvalues[0] if len(eigenvalues) else 0.0
    numerical_rank = int(np.sum(eigenvalues > largest * 1e-6)) if largest > 0 else 0
    metrics = {
        'num_clips': int(embeddings.shape[0]),
        'num_embeddings': int(flat.shape[0]),
        'embedding_dim': int(flat.shape[-1]),
        'sigreg_loss_mean': float(np.mean(sigreg_values)),
        'sigreg_loss_std': float(np.std(sigreg_values)),
        'feature_mean_abs_mean': float(np.mean(np.abs(feature_mean))),
        'feature_mean_abs_max': float(np.max(np.abs(feature_mean))),
        'feature_std_mean': float(np.mean(feature_std)),
        'feature_std_min': float(np.min(feature_std)),
        'feature_std_max': float(np.max(feature_std)),
        'covariance_eigenvalue_max': float(largest),
        'covariance_eigenvalue_min': float(eigenvalues[-1]),
        'effective_rank_entropy': effective_rank,
        'participation_ratio': participation_ratio,
        'numerical_rank_1e-6': numerical_rank,
        'pairwise_distance_mean': float(np.mean(distances)),
        'pairwise_distance_std': float(np.std(distances)),
        'pairwise_distance_p05': float(np.quantile(distances, 0.05)),
        'pairwise_distance_p95': float(np.quantile(distances, 0.95)),
    }
    if prediction_losses:
        metrics['prediction_loss_mean'] = float(np.mean(prediction_losses))
        metrics['prediction_loss_std'] = float(np.std(prediction_losses))
    return metrics


def _batch_stats_delta(before, after):
    before_leaves = jax.tree_util.tree_leaves(before)
    after_leaves = jax.tree_util.tree_leaves(after)
    if len(before_leaves) != len(after_leaves):
        raise ValueError('Batch-statistic trees differ after calibration.')
    squared_difference = 0.0
    squared_before = 0.0
    max_absolute_difference = 0.0
    num_values = 0
    for before_leaf, after_leaf in zip(before_leaves, after_leaves):
        before_array = np.asarray(jax.device_get(before_leaf), dtype=np.float64)
        after_array = np.asarray(jax.device_get(after_leaf), dtype=np.float64)
        difference = after_array - before_array
        squared_difference += float(np.square(difference).sum())
        squared_before += float(np.square(before_array).sum())
        max_absolute_difference = max(
            max_absolute_difference, float(np.max(np.abs(difference)))
        )
        num_values += before_array.size
    return {
        'num_values': int(num_values),
        'l2_difference': float(np.sqrt(squared_difference)),
        'relative_l2_difference': float(
            np.sqrt(squared_difference / max(squared_before, 1e-30))
        ),
        'max_absolute_difference': max_absolute_difference,
    }


def _save_checkpoint(source, destination, calibrated_batch_stats, calibration):
    payload = flax.serialization.msgpack_restore(Path(source).read_bytes())
    payload['batch_stats'] = jax.device_get(calibrated_batch_stats)
    payload['bn_calibration'] = calibration
    serialized = flax.serialization.msgpack_serialize(payload)
    temporary = destination.with_suffix(destination.suffix + '.tmp')
    with temporary.open('wb') as file:
        file.write(serialized)
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary, destination)


def main():
    args = parse_args()
    if args.batch_size <= 1:
        raise ValueError('--batch-size must be greater than one for BatchNorm.')
    if args.sample_batches <= 0 or args.calibration_batches <= 0:
        raise ValueError('Batch counts must be positive.')

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model, variables, metadata = load_frozen_lewm(args.checkpoint)
    config = metadata['config']
    dataset = LeWMNPZSequenceDataset(
        args.dataset_path,
        args.validation_dataset_path,
        num_steps=int(config['history_size']) + int(config.get('num_preds', 1)),
        frameskip=int(config['frameskip']),
        seed=args.seed,
    )
    rng = np.random.default_rng(args.seed)
    train_batches = _sample_batches(
        dataset,
        dataset.train_indices,
        batch_size=args.batch_size,
        num_batches=args.sample_batches,
        rng=rng,
    )
    validation_batches = _sample_batches(
        dataset,
        dataset.val_indices,
        batch_size=args.batch_size,
        num_batches=args.sample_batches,
        rng=rng,
    )

    @jax.jit
    def inference_step(params, batch_stats, pixels, actions):
        return model.apply(
            {'params': params, 'batch_stats': batch_stats},
            pixels,
            actions,
            train=False,
        )

    @jax.jit
    def batch_stat_encode_step(params, batch_stats, pixels):
        embeddings, _ = model.apply(
            {'params': params, 'batch_stats': batch_stats},
            pixels,
            train=True,
            method=model.encode_pixels,
            mutable=['batch_stats'],
        )
        return embeddings

    @jax.jit
    def calibration_step(params, batch_stats, pixels, actions, dropout_key):
        _, updates = model.apply(
            {'params': params, 'batch_stats': batch_stats},
            pixels,
            actions,
            train=True,
            rngs={'dropout': dropout_key},
            mutable=['batch_stats'],
        )
        return updates['batch_stats']

    @jax.jit
    def sigreg_step(embeddings, key):
        return sigreg_loss(
            embeddings,
            key,
            knots=int(config.get('sigreg_knots', 17)),
            num_proj=int(config.get('sigreg_num_proj', 1024)),
        )

    params = variables['params']
    original_batch_stats = variables['batch_stats']
    key = jax.random.PRNGKey(args.seed)

    def evaluate(batches, batch_stats, *, use_current_batch_stats):
        embedding_batches = []
        prediction_losses = []
        for batch in batches:
            pixels = jnp.asarray(batch['pixels'])
            actions = jnp.asarray(batch['action'])
            if use_current_batch_stats:
                embeddings = batch_stat_encode_step(
                    params, batch_stats, pixels
                )
            else:
                embeddings, predictions = inference_step(
                    params, batch_stats, pixels, actions
                )
                targets = embeddings[:, 1:]
                prediction_losses.append(
                    float(jax.device_get(jnp.mean((predictions - targets) ** 2)))
                )
            embedding_batches.append(np.asarray(jax.device_get(embeddings)))
        return _geometry_metrics(
            embedding_batches, sigreg_step, key, prediction_losses
        )

    results = {
        'original_running_train': evaluate(
            train_batches, original_batch_stats, use_current_batch_stats=False
        ),
        'original_running_validation': evaluate(
            validation_batches, original_batch_stats, use_current_batch_stats=False
        ),
        'current_batch_train': evaluate(
            train_batches, original_batch_stats, use_current_batch_stats=True
        ),
        'current_batch_validation': evaluate(
            validation_batches, original_batch_stats, use_current_batch_stats=True
        ),
    }

    calibrated_batch_stats = original_batch_stats
    shuffled = dataset.shuffled_train_indices()
    required = args.calibration_batches * args.batch_size
    if required > len(shuffled):
        repeats = (required + len(shuffled) - 1) // len(shuffled)
        shuffled = np.tile(shuffled, repeats)
    for batch_index in range(args.calibration_batches):
        start = batch_index * args.batch_size
        batch = dataset.get_batch(shuffled[start : start + args.batch_size])
        dropout_key = jax.random.fold_in(key, batch_index)
        calibrated_batch_stats = calibration_step(
            params,
            calibrated_batch_stats,
            jnp.asarray(batch['pixels']),
            jnp.asarray(batch['action']),
            dropout_key,
        )
        if (batch_index + 1) % 25 == 0 or batch_index + 1 == args.calibration_batches:
            print(
                f'Calibrated {batch_index + 1}/{args.calibration_batches} batches',
                flush=True,
            )

    results['calibrated_running_train'] = evaluate(
        train_batches, calibrated_batch_stats, use_current_batch_stats=False
    )
    results['calibrated_running_validation'] = evaluate(
        validation_batches, calibrated_batch_stats, use_current_batch_stats=False
    )
    batch_stats_delta = _batch_stats_delta(
        original_batch_stats, calibrated_batch_stats
    )
    calibration = {
        'source_checkpoint': str(Path(args.checkpoint).resolve()),
        'dataset_path': str(Path(args.dataset_path).resolve()),
        'validation_dataset_path': str(
            Path(args.validation_dataset_path).resolve()
        ),
        'batch_size': args.batch_size,
        'calibration_batches': args.calibration_batches,
        'seed': args.seed,
    }
    checkpoint_path = (
        output_dir
        / f'weights_epoch_{metadata["epoch"]}_bncalib{args.calibration_batches}.msgpack'
    )
    _save_checkpoint(
        args.checkpoint,
        checkpoint_path,
        calibrated_batch_stats,
        calibration,
    )
    report = {
        'checkpoint': metadata,
        'device': str(jax.devices()[0]),
        'batch_size': args.batch_size,
        'sample_batches': args.sample_batches,
        'calibration_batches': args.calibration_batches,
        'batch_stats_delta': batch_stats_delta,
        'results': results,
        'calibrated_checkpoint': str(checkpoint_path.resolve()),
    }
    report_path = output_dir / 'bn_geometry_diagnostic.json'
    with report_path.open('w') as file:
        json.dump(report, file, indent=2)
        file.write('\n')
    dataset.close()
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
