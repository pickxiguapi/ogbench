"""Train an ACID-style flow-matching IDM on frozen LeWM latent transitions."""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training import train_state

from acid_idm import ARCHITECTURE, ACIDInverseDynamicsFlow, sample_inverse_actions
from utils.latent_subgoal_dataset import split_episodes


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--latent-dataset', required=True)
    parser.add_argument('--save-dir', required=True)
    parser.add_argument('--exp-name', required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--split-seed', type=int, default=0)
    parser.add_argument('--train-fraction', type=float, default=0.9)
    parser.add_argument('--transition-steps', type=int, default=5)
    parser.add_argument('--train-steps', type=int, default=200_000)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--model-dim', type=int, default=192)
    parser.add_argument('--num-layers', type=int, default=4)
    parser.add_argument('--num-heads', type=int, default=3)
    parser.add_argument('--mlp-dim', type=int, default=768)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--final-learning-rate', type=float, default=1e-6)
    parser.add_argument('--warmup-steps', type=int, default=2000)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--gradient-clip', type=float, default=1.0)
    parser.add_argument('--validation-pairs', type=int, default=50_000)
    parser.add_argument('--eval-batch-size', type=int, default=5000)
    parser.add_argument('--log-interval', type=int, default=1000)
    parser.add_argument('--eval-interval', type=int, default=5000)
    parser.add_argument('--checkpoint-interval', type=int, default=25_000)
    parser.add_argument('--resume', action='store_true')
    return parser.parse_args()


def validate_args(args):
    for name in (
        'transition_steps',
        'train_steps',
        'batch_size',
        'model_dim',
        'num_layers',
        'num_heads',
        'mlp_dim',
        'warmup_steps',
        'validation_pairs',
        'eval_batch_size',
        'log_interval',
        'eval_interval',
        'checkpoint_interval',
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f'{name} must be positive.')
    if not 0.0 < args.train_fraction < 1.0:
        raise ValueError('train_fraction must be in (0, 1).')
    if args.warmup_steps >= args.train_steps:
        raise ValueError('warmup_steps must be smaller than train_steps.')
    if args.model_dim % args.num_heads:
        raise ValueError('model_dim must be divisible by num_heads.')
    if args.model_dim % 2:
        raise ValueError('model_dim must be even for sinusoidal time embeddings.')


def json_safe(value):
    if isinstance(value, jax.Array):
        return np.asarray(value).tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def write_json_atomic(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w') as file:
        json.dump(json_safe(value), file, indent=2, sort_keys=True)
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary, path)


def append_jsonl(path, value):
    with Path(path).open('a') as file:
        file.write(json.dumps(json_safe(value), sort_keys=True) + '\n')
        file.flush()


def load_dataset(path):
    import h5py

    path = Path(path).expanduser().resolve()
    with h5py.File(path, 'r') as file:
        if file.attrs.get('format') != 'lewm_latent_dataset':
            raise ValueError(f'Not a LeWM latent cache: {path}')
        if file.attrs.get('status') != 'complete':
            raise ValueError(f'Incomplete LeWM latent cache: {path}')
        z = np.asarray(file['z'], dtype=np.float32)
        actions = np.asarray(file['action'], dtype=np.float32)
        offsets = np.asarray(file['ep_offset'], dtype=np.int64)
        lengths = np.asarray(file['ep_len'], dtype=np.int64)
        metadata = {key: value for key, value in file.attrs.items()}
    if len(z) != len(actions) or z.ndim != 2 or actions.ndim != 2:
        raise ValueError('Latents and actions must be aligned rank-two arrays.')
    if not np.isfinite(z).all():
        raise FloatingPointError('Latent cache contains non-finite values.')
    return path, z, actions, offsets, lengths, metadata


def build_transition_starts(offsets, lengths, episodes, transition_steps):
    groups = []
    for episode in np.asarray(episodes, dtype=np.int64):
        offset = int(offsets[episode])
        length = int(lengths[episode])
        count = length - int(transition_steps)
        if count > 0:
            groups.append(np.arange(offset, offset + count, dtype=np.int32))
    if not groups:
        raise ValueError('Selected episodes contain no complete transitions.')
    return np.concatenate(groups)


def action_statistics(actions, offsets, lengths, train_episodes):
    mask = np.zeros(len(actions), dtype=bool)
    for episode in np.asarray(train_episodes, dtype=np.int64):
        offset = int(offsets[episode])
        length = int(lengths[episode])
        # The last observation has no valid outgoing action.
        mask[offset : offset + max(length - 1, 0)] = True
    values = actions[mask]
    values = values[np.isfinite(values).all(axis=1)]
    if not len(values):
        raise ValueError('Training split contains no finite actions.')
    mean = values.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = values.std(axis=0, ddof=1, dtype=np.float64).astype(np.float32)
    std = np.where(std > 1e-6, std, 1.0).astype(np.float32)
    return mean, std


def checkpoint_path(output_dir, step):
    return Path(output_dir) / f'checkpoint_{int(step):06d}.msgpack'


def save_checkpoint(state, rng, output_dir, config, action_mean, action_std):
    step = int(jax.device_get(state.step))
    payload = {
        'step': step,
        'config': config,
        'params': jax.device_get(state.params),
        'action_mean': np.asarray(action_mean),
        'action_std': np.asarray(action_std),
        'train_state': flax.serialization.to_state_dict(jax.device_get(state)),
        'rng': np.asarray(jax.device_get(rng)),
    }
    path = checkpoint_path(output_dir, step)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('wb') as file:
        file.write(flax.serialization.msgpack_serialize(payload))
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary, path)
    return path


def latest_checkpoint(output_dir):
    candidates = []
    for path in Path(output_dir).glob('checkpoint_*.msgpack'):
        match = re.fullmatch(r'checkpoint_(\d+)\.msgpack', path.name)
        if match:
            candidates.append((int(match.group(1)), path))
    return max(candidates, default=(None, None))


def restore_checkpoint(state, path):
    payload = flax.serialization.msgpack_restore(Path(path).read_bytes())
    state = flax.serialization.from_state_dict(state, payload['train_state'])
    rng = jnp.asarray(payload['rng'], dtype=jnp.uint32)
    if int(state.step) != int(payload['step']):
        raise ValueError(f'Checkpoint step mismatch: {path}')
    return state, rng


def main():
    args = parse_args()
    validate_args(args)
    output_dir = Path(args.save_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / 'metrics.jsonl'

    dataset_path, z, actions, offsets, lengths, metadata = load_dataset(
        args.latent_dataset
    )
    train_episodes, val_episodes = split_episodes(
        len(offsets), args.train_fraction, args.split_seed
    )
    train_starts = build_transition_starts(
        offsets, lengths, train_episodes, args.transition_steps
    )
    val_starts = build_transition_starts(
        offsets, lengths, val_episodes, args.transition_steps
    )
    action_mean, action_std = action_statistics(
        actions, offsets, lengths, train_episodes
    )
    embed_dim = int(z.shape[1])
    atomic_action_dim = int(actions.shape[1])
    action_chunk_dim = atomic_action_dim * args.transition_steps
    config = {
        **vars(args),
        'architecture': ARCHITECTURE,
        'latent_dataset': str(dataset_path),
        'task': str(metadata.get('task', '')),
        'lewm_checkpoint_sha256': str(metadata.get('checkpoint_sha256', '')),
        'embed_dim': embed_dim,
        'atomic_action_dim': atomic_action_dim,
        'action_chunk_dim': action_chunk_dim,
        'flow_time_distribution': 'beta_1.5_1.0',
        'flow_path': 'x_tau=tau*noise+(1-tau)*action',
        'loss': 'conditional_flow_matching_mse',
        'inference_solver': 'reverse_euler',
        'inference_steps': 1,
    }
    config.pop('resume')
    config_path = output_dir / 'config.json'
    if config_path.exists():
        existing = json.loads(config_path.read_text())
        if existing != json_safe(config):
            raise ValueError(f'Existing config differs: {config_path}')
    else:
        write_json_atomic(config_path, config)

    rng = np.random.default_rng(args.split_seed + 1)
    replace = len(val_starts) < args.validation_pairs
    fixed_val = rng.choice(
        val_starts, size=args.validation_pairs, replace=replace
    ).astype(np.int32)
    print(
        f'dataset={dataset_path} rows={len(z)} episodes={len(offsets)} '
        f'train_transitions={len(train_starts)} val_transitions={len(val_starts)} '
        f'embed_dim={embed_dim} action_chunk_dim={action_chunk_dim}',
        flush=True,
    )
    print(f'JAX backend={jax.default_backend()} devices={jax.devices()}', flush=True)

    z_device = jax.device_put(z)
    actions_device = jax.device_put(actions)
    train_starts_device = jax.device_put(train_starts)
    action_mean_device = jax.device_put(action_mean)
    action_std_device = jax.device_put(action_std)
    action_offsets = jnp.arange(args.transition_steps, dtype=jnp.int32)

    model = ACIDInverseDynamicsFlow(
        embed_dim=embed_dim,
        action_dim=action_chunk_dim,
        model_dim=args.model_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        mlp_dim=args.mlp_dim,
    )
    init_key, train_key = jax.random.split(jax.random.PRNGKey(args.seed))
    variables = model.init(
        init_key,
        jnp.zeros((1, action_chunk_dim), dtype=jnp.float32),
        jnp.zeros((1, embed_dim), dtype=jnp.float32),
        jnp.zeros((1, embed_dim), dtype=jnp.float32),
        jnp.ones((1,), dtype=jnp.float32),
    )
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=args.learning_rate,
        warmup_steps=args.warmup_steps,
        decay_steps=args.train_steps,
        end_value=args.final_learning_rate,
    )
    decay_mask = jax.tree_util.tree_map(
        lambda value: value.ndim > 1, variables['params']
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(args.gradient_clip),
        optax.adamw(
            schedule,
            b1=0.9,
            b2=0.999,
            weight_decay=args.weight_decay,
            mask=decay_mask,
        ),
    )
    state = train_state.TrainState.create(
        apply_fn=model.apply, params=variables['params'], tx=optimizer
    )
    print(
        'Model parameters='
        f'{sum(value.size for value in jax.tree_util.tree_leaves(state.params)):,}',
        flush=True,
    )

    if args.resume:
        _, path = latest_checkpoint(output_dir)
        if path is not None:
            state, train_key = restore_checkpoint(state, path)
            print(f'Resumed {path}', flush=True)

    @jax.jit
    def train_step(state, key):
        key, index_key, noise_key, time_key = jax.random.split(key, 4)
        positions = jax.random.randint(
            index_key, (args.batch_size,), 0, len(train_starts_device)
        )
        starts = train_starts_device[positions]
        current = z_device[starts]
        next_z = z_device[starts + args.transition_steps]
        rows = starts[:, None] + action_offsets[None]
        chunks = (actions_device[rows] - action_mean_device) / action_std_device
        chunks = jnp.nan_to_num(chunks).reshape(args.batch_size, action_chunk_dim)
        noise = jax.random.normal(noise_key, chunks.shape, dtype=jnp.float32)
        tau = jax.random.beta(
            time_key, 1.5, 1.0, shape=(args.batch_size,), dtype=jnp.float32
        )
        interpolation = tau[:, None] * noise + (1.0 - tau[:, None]) * chunks
        target_velocity = noise - chunks

        def loss_fn(params):
            prediction = model.apply(
                {'params': params}, interpolation, current, next_z, tau
            )
            error = prediction - target_velocity
            loss = jnp.mean(jnp.square(error))
            return loss, {
                'flow_matching_mse': loss,
                'velocity_norm': jnp.mean(jnp.linalg.norm(prediction, axis=-1)),
                'target_velocity_norm': jnp.mean(
                    jnp.linalg.norm(target_velocity, axis=-1)
                ),
            }

        (_, metrics), gradients = jax.value_and_grad(loss_fn, has_aux=True)(
            state.params
        )
        state = state.apply_gradients(grads=gradients)
        metrics['gradient_norm'] = optax.global_norm(gradients)
        metrics['learning_rate'] = schedule(state.step)
        return state, key, metrics

    @jax.jit
    def predict_actions(params, current, next_z, key):
        return sample_inverse_actions(
            model, params, current, next_z, key, num_steps=1
        )

    def evaluate(params, seed):
        example_mse = []
        example_mae = []
        eval_key = jax.random.PRNGKey(seed)
        for start in range(0, len(fixed_val), args.eval_batch_size):
            indices = fixed_val[start : start + args.eval_batch_size]
            current = z_device[indices]
            next_z = z_device[indices + args.transition_steps]
            rows = jnp.asarray(indices)[:, None] + action_offsets[None]
            targets = (
                (actions_device[rows] - action_mean_device) / action_std_device
            ).reshape(len(indices), action_chunk_dim)
            eval_key, batch_key = jax.random.split(eval_key)
            predictions = predict_actions(params, current, next_z, batch_key)
            errors = np.asarray(jax.device_get(predictions - targets))
            example_mse.append(np.mean(np.square(errors), axis=-1))
            example_mae.append(np.mean(np.abs(errors), axis=-1))
        example_mse = np.concatenate(example_mse)
        example_mae = np.concatenate(example_mae)
        metrics = {
            'inverse_action_mse': float(example_mse.mean()),
            'inverse_action_mae': float(example_mae.mean()),
        }
        for quantile in (0.50, 0.80, 0.90, 0.95, 0.99):
            suffix = int(round(100 * quantile))
            metrics[f'inverse_action_mse_q{suffix}'] = float(
                np.quantile(example_mse, quantile)
            )
        return metrics

    def record_validation(step, metrics):
        row = {'type': 'validation', 'step': step, **metrics}
        append_jsonl(metrics_path, row)
        write_json_atomic(
            output_dir / 'calibration.json',
            {
                **row,
                'split': 'heldout_episodes',
                'score': 'mean squared normalized action residual',
                'validation_pairs': args.validation_pairs,
                'evaluation_seed': args.split_seed + 2,
            },
        )

    current_step = int(state.step)
    started = time.monotonic()
    initial_metrics = evaluate(state.params, args.split_seed + 2)
    record_validation(current_step, initial_metrics)
    print(f'Validation step={current_step}: {json.dumps(initial_metrics)}', flush=True)
    latest_metrics = None
    while current_step < args.train_steps:
        state, train_key, latest_metrics = train_step(state, train_key)
        current_step = int(state.step)
        if current_step % args.log_interval == 0 or current_step == args.train_steps:
            row = {
                'type': 'train',
                'step': current_step,
                'elapsed_seconds': time.monotonic() - started,
                **jax.device_get(latest_metrics),
            }
            append_jsonl(metrics_path, row)
            print(f'Train step={current_step}: {json.dumps(json_safe(row), sort_keys=True)}', flush=True)
        if current_step % args.eval_interval == 0 or current_step == args.train_steps:
            metrics = evaluate(state.params, args.split_seed + 2)
            record_validation(current_step, metrics)
            print(f'Validation step={current_step}: {json.dumps(metrics, sort_keys=True)}', flush=True)
        if current_step % args.checkpoint_interval == 0 or current_step == args.train_steps:
            path = save_checkpoint(
                state, train_key, output_dir, config, action_mean, action_std
            )
            print(f'Saved checkpoint: {path}', flush=True)

    write_json_atomic(
        output_dir / 'complete.json',
        {
            'step': current_step,
            'checkpoint': str(checkpoint_path(output_dir, current_step)),
            'elapsed_seconds': time.monotonic() - started,
            'last_train_metrics': jax.device_get(latest_metrics),
        },
    )


if __name__ == '__main__':
    main()
