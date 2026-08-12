"""Train the JAX LeWM reproduction on an existing LeWM Lance dataset."""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training import train_state
from lewm_jax import (
    REFERENCE_ARCHITECTURE,
    architecture_for_encoder,
    build_model,
    loss_for_architecture,
    uses_imagenet_preprocessing,
)
from utils.lewm_sequence_dataset import LeWMSequenceDataset


@dataclass(frozen=True)
class LeWMConfig:
    architecture: str = REFERENCE_ARCHITECTURE
    encoder: str = 'vit_tiny14'
    seed: int = 3072
    epochs: int = 10
    batch_size: int = 128
    decode_workers: int = 6
    train_fraction: float = 0.9
    image_size: int = 224
    embed_dim: int = 192
    patch_size: int = 14
    history_size: int = 3
    num_preds: int = 1
    frameskip: int = 5
    projector_hidden_dim: int = 2048
    action_smoothed_dim: int = 10
    action_mlp_scale: int = 4
    predictor_depth: int = 6
    predictor_heads: int = 16
    predictor_dim_head: int = 64
    predictor_mlp_dim: int = 2048
    predictor_dropout: float = 0.1
    predictor_emb_dropout: float = 0.0
    learning_rate: float = 5e-5
    lr_schedule: str = 'optimizer_step_warmup_cosine_1pct'
    weight_decay: float = 1e-3
    gradient_clip: float = 1.0
    sigreg_weight: float = 0.09
    sigreg_knots: int = 17
    sigreg_num_proj: int = 1024
    precision: str = 'bf16'


class LeWMTrainState(train_state.TrainState):
    batch_stats: flax.core.FrozenDict


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset_path', required=True)
    parser.add_argument('--save_dir', required=True)
    parser.add_argument('--exp_name', required=True)
    parser.add_argument('--decode_workers', type=int, default=6)
    parser.add_argument('--encoder', choices=('vit_tiny14', 'impala_small'), default='vit_tiny14')
    parser.add_argument('--seed', type=int, default=3072)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--frameskip', type=int, default=5)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    parser.add_argument('--sigreg_weight', type=float, default=0.09)
    parser.add_argument('--sigreg_knots', type=int, default=17)
    parser.add_argument('--sigreg_num_proj', type=int, default=1024)
    return parser.parse_args()


def warmup_cosine_schedule(base_lr, total_steps):
    """Standard optimizer-step warmup followed by cosine decay."""
    warmup_steps = max(1, int(0.01 * total_steps))
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=base_lr,
        warmup_steps=warmup_steps,
        decay_steps=total_steps,
        end_value=0.0,
    )
    return schedule, warmup_steps


def make_steps(model, loss_function, learning_rate_schedule, config):
    @jax.jit
    def train_step(state, batch, dropout_key, sigreg_key):
        def loss_fn(params):
            variables = {'params': params, 'batch_stats': state.batch_stats}
            return loss_function(
                model,
                variables,
                batch,
                train=True,
                dropout_key=dropout_key,
                sigreg_key=sigreg_key,
                sigreg_weight=config.sigreg_weight,
                sigreg_knots=config.sigreg_knots,
                sigreg_num_proj=config.sigreg_num_proj,
            )

        (_, (metrics, batch_stats)), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        state = state.apply_gradients(grads=grads).replace(batch_stats=batch_stats)
        metrics = dict(metrics)
        metrics['grad_norm'] = optax.global_norm(grads)
        metrics['learning_rate'] = learning_rate_schedule(state.step - 1)
        return state, metrics

    @jax.jit
    def validation_step(state, batch, dropout_key, sigreg_key):
        variables = {'params': state.params, 'batch_stats': state.batch_stats}
        _, (metrics, _) = loss_function(
            model,
            variables,
            batch,
            train=False,
            dropout_key=dropout_key,
            sigreg_key=sigreg_key,
            sigreg_weight=config.sigreg_weight,
            sigreg_knots=config.sigreg_knots,
            sigreg_num_proj=config.sigreg_num_proj,
        )
        return metrics

    return train_step, validation_step


def mean_metrics(rows):
    return {key: float(np.mean([float(row[key]) for row in rows])) for key in rows[0]}


def save_model(state, path, epoch, config):
    payload = {
        'params': jax.device_get(state.params),
        'batch_stats': jax.device_get(state.batch_stats),
        'epoch': epoch,
        'config': asdict(config),
    }
    serialized = flax.serialization.msgpack_serialize(payload)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('wb') as file:
        file.write(serialized)
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary, path)


def main():
    args = parse_args()
    config = LeWMConfig(
        architecture=architecture_for_encoder(args.encoder),
        encoder=args.encoder,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        decode_workers=args.decode_workers,
        frameskip=args.frameskip,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        sigreg_weight=args.sigreg_weight,
        sigreg_knots=args.sigreg_knots,
        sigreg_num_proj=args.sigreg_num_proj,
    )
    output_dir = Path(args.save_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / 'config.json').open('w') as file:
        json.dump({'exp_name': args.exp_name, 'dataset_path': args.dataset_path, **asdict(config)}, file, indent=2)

    dataset = LeWMSequenceDataset(
        args.dataset_path,
        num_steps=config.history_size + config.num_preds,
        frameskip=config.frameskip,
        train_fraction=config.train_fraction,
        seed=config.seed,
        decode_workers=config.decode_workers,
        normalize_pixels=uses_imagenet_preprocessing(config),
    )
    steps_per_epoch = len(dataset.train_indices) // config.batch_size
    if steps_per_epoch < 1:
        raise ValueError('The training split is smaller than one full batch.')
    total_steps = config.epochs * steps_per_epoch
    lr_schedule, warmup_steps = warmup_cosine_schedule(config.learning_rate, total_steps)
    optimizer = optax.chain(
        optax.clip_by_global_norm(config.gradient_clip),
        optax.adamw(lr_schedule, weight_decay=config.weight_decay),
    )

    model = build_model(config, dtype=jnp.bfloat16)
    loss_function = loss_for_architecture(config.architecture)
    rng = jax.random.PRNGKey(config.seed)
    rng, params_key, dropout_key = jax.random.split(rng, 3)
    example = dataset.get_batch(dataset.train_indices[:2])
    variables = model.init(
        {'params': params_key, 'dropout': dropout_key},
        jnp.asarray(example['pixels']),
        jnp.asarray(example['action']),
        # PyTorch constructs BatchNorm with running mean=0 and variance=1;
        # neither module initialization nor Lightning's validation sanity
        # check mutates those statistics before the first training batch.
        # Initializing Flax in training mode would otherwise consume this
        # two-example shape probe as an extra BatchNorm update.
        train=False,
    )
    state = LeWMTrainState.create(
        apply_fn=model.apply,
        params=variables['params'],
        batch_stats=variables['batch_stats'],
        tx=optimizer,
    )
    train_step, validation_step = make_steps(
        model, loss_function, lr_schedule, config
    )
    parameter_count = sum(value.size for value in jax.tree_util.tree_leaves(state.params))

    print(f'exp_name={args.exp_name}')
    print(f'encoder={config.encoder}')
    print(f'dataset={args.dataset_path}')
    print(f'clips train={len(dataset.train_indices)} val={len(dataset.val_indices)}')
    print(f'steps_per_epoch={steps_per_epoch} total_steps={total_steps} scheduler_warmup_steps={warmup_steps}')
    print(f'parameters={parameter_count:,} devices={jax.devices()}')

    csv_path = output_dir / 'metrics.csv'
    fieldnames = [
        'epoch',
        'train_loss',
        'train_pred_loss',
        'train_sigreg_loss',
        'train_grad_norm',
        'learning_rate',
        'val_loss',
        'val_pred_loss',
        'val_sigreg_loss',
        'epoch_seconds',
        'total_seconds',
    ]
    start_time = time.time()
    with csv_path.open('w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for epoch in range(1, config.epochs + 1):
            epoch_start = time.time()
            shuffled = dataset.shuffled_train_indices()
            shuffled = shuffled[: steps_per_epoch * config.batch_size]
            train_rows = []
            for start in range(0, len(shuffled), config.batch_size):
                batch_np = dataset.get_batch(shuffled[start : start + config.batch_size])
                batch = jax.tree_util.tree_map(jnp.asarray, batch_np)
                rng, dropout_key, sigreg_key = jax.random.split(rng, 3)
                state, metrics = train_step(state, batch, dropout_key, sigreg_key)
                train_rows.append(jax.device_get(metrics))

            val_rows = []
            for start in range(0, len(dataset.val_indices), config.batch_size):
                indices = dataset.val_indices[start : start + config.batch_size]
                if not len(indices):
                    continue
                batch_np = dataset.get_batch(indices)
                batch = jax.tree_util.tree_map(jnp.asarray, batch_np)
                rng, dropout_key, sigreg_key = jax.random.split(rng, 3)
                val_rows.append(jax.device_get(validation_step(state, batch, dropout_key, sigreg_key)))

            train_metrics = mean_metrics(train_rows)
            val_metrics = mean_metrics(val_rows)
            row = {
                'epoch': epoch,
                'train_loss': train_metrics['loss'],
                'train_pred_loss': train_metrics['pred_loss'],
                'train_sigreg_loss': train_metrics['sigreg_loss'],
                'train_grad_norm': train_metrics['grad_norm'],
                'learning_rate': train_metrics['learning_rate'],
                'val_loss': val_metrics['loss'],
                'val_pred_loss': val_metrics['pred_loss'],
                'val_sigreg_loss': val_metrics['sigreg_loss'],
                'epoch_seconds': time.time() - epoch_start,
                'total_seconds': time.time() - start_time,
            }
            writer.writerow(row)
            csv_file.flush()
            print(json.dumps(row))
            save_model(state, output_dir / f'weights_epoch_{epoch}.msgpack', epoch, config)

    dataset.close()


if __name__ == '__main__':
    main()
