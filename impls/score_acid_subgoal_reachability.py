"""Score traced LeWM++ subgoals with ACID consistency and real reachability."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from acid_idm import load_acid_idm_checkpoint
from lewm_jax.checkpoints import load_frozen_lewm


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', required=True)
    parser.add_argument('--trace-dir', required=True)
    parser.add_argument('--lewm-checkpoint', required=True)
    parser.add_argument('--idm-checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--real-horizon', type=int, default=10)
    parser.add_argument('--transition-steps', type=int, default=5)
    parser.add_argument('--encode-batch-size', type=int, default=256)
    parser.add_argument('--seed', type=int, default=0)
    return parser.parse_args()


def mse(left, right):
    return np.mean(np.square(np.asarray(left) - np.asarray(right)), axis=-1)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def safe_mean(values):
    values = np.asarray(values, dtype=np.float64)
    return float(np.nanmean(values)) if len(values) else float('nan')


def safe_std(values):
    values = np.asarray(values, dtype=np.float64)
    return float(np.nanstd(values)) if len(values) else float('nan')


def strict_json(value):
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: strict_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [strict_json(item) for item in value]
    return value


def average_ranks(values):
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind='mergesort')
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1)
        start = stop
    return ranks


def binary_auc(labels, scores):
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=np.float64)
    finite = np.isfinite(scores)
    labels = labels[finite]
    scores = scores[finite]
    positives = int(labels.sum())
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return float('nan')
    ranks = average_ranks(scores)
    return float(
        (ranks[labels].sum() - positives * (positives - 1) / 2)
        / (positives * negatives)
    )


def correlation(left, right, *, rank=False):
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    finite = np.isfinite(left) & np.isfinite(right)
    left = left[finite]
    right = right[finite]
    if len(left) < 2:
        return float('nan')
    if rank:
        left = average_ranks(left)
        right = average_ranks(right)
    if np.std(left) == 0 or np.std(right) == 0:
        return float('nan')
    return float(np.corrcoef(left, right)[0, 1])


def main():
    args = parse_args()
    if args.real_horizon <= 0 or args.transition_steps <= 0:
        raise ValueError('Horizons must be positive.')
    trace_paths = sorted(Path(args.trace_dir).glob('episode_*.npz'))
    if not trace_paths:
        raise FileNotFoundError(f'No trace files in {args.trace_dir}')

    lewm, lewm_variables, lewm_metadata = load_frozen_lewm(args.lewm_checkpoint)
    idm, idm_params, action_mean, action_std, idm_config, idm_step = (
        load_acid_idm_checkpoint(args.idm_checkpoint)
    )
    if int(idm_config['transition_steps']) != args.transition_steps:
        raise ValueError('IDM transition length does not match requested scoring step.')
    lewm_sha256 = sha256_file(args.lewm_checkpoint)
    if idm_config['lewm_checkpoint_sha256'] != lewm_sha256:
        raise ValueError(
            'IDM was trained with a different frozen LeWM checkpoint: '
            f'{idm_config["lewm_checkpoint_sha256"]} != {lewm_sha256}'
        )
    if int(lewm_metadata['config']['embed_dim']) != int(idm_config['embed_dim']):
        raise ValueError('LeWM and IDM embedding dimensions differ.')
    atomic_action_dim = int(idm_config['atomic_action_dim'])
    action_mean = np.asarray(action_mean, dtype=np.float32)
    action_std = np.asarray(action_std, dtype=np.float32)
    if action_mean.shape != (atomic_action_dim,) or action_std.shape != (
        atomic_action_dim,
    ):
        raise ValueError('IDM action statistics have unexpected shapes.')

    encode = jax.jit(
        lambda pixels: lewm.apply(
            lewm_variables, pixels, train=False, method=lewm.encode_pixels
        ).astype(jnp.float32)
    )

    def encode_frames(frames):
        outputs = []
        for start in range(0, len(frames), args.encode_batch_size):
            outputs.append(
                np.asarray(
                    jax.device_get(
                        encode(jnp.asarray(frames[start : start + args.encode_batch_size]))
                    )
                )
            )
        return np.concatenate(outputs)

    @jax.jit
    def inverse_one_step(current, next_z, keys):
        noise = jax.vmap(
            lambda key: jax.random.normal(
                key, (int(idm.action_dim),), dtype=jnp.float32
            )
        )(keys)
        tau = jnp.ones((len(current),), dtype=jnp.float32)
        velocity = idm.apply(
            {'params': idm_params}, noise, current, next_z, tau
        )
        return noise - velocity

    event_rows = []
    acid_current = []
    acid_next = []
    acid_actions = []
    acid_keys = []
    acid_event_indices = []
    base_key = jax.random.PRNGKey(args.seed)

    for episode_index, trace_path in enumerate(trace_paths):
        with np.load(trace_path, allow_pickle=False) as trace:
            required = {
                'frames',
                'success',
                'plan_steps',
                'current_embeddings',
                'predicted_paths',
                'imagined_paths',
                'environment_action_blocks',
            }
            missing = required.difference(trace.files)
            if missing:
                raise ValueError(f'{trace_path} is missing {sorted(missing)}')
            frames = np.asarray(trace['frames'])
            frame_latents = encode_frames(frames)
            success = bool(trace['success'])
            plan_steps = np.asarray(trace['plan_steps'], dtype=np.int32)
            current_embeddings = np.asarray(
                trace['current_embeddings'], dtype=np.float32
            )
            predicted_paths = np.asarray(trace['predicted_paths'], dtype=np.float32)
            imagined_paths = np.asarray(trace['imagined_paths'], dtype=np.float32)
            environment_blocks = np.asarray(
                trace['environment_action_blocks'], dtype=np.float32
            )
        event_count = len(plan_steps)
        if not (
            len(current_embeddings)
            == len(predicted_paths)
            == len(imagined_paths)
            == len(environment_blocks)
            == event_count
        ):
            raise ValueError(f'Trace event arrays are misaligned: {trace_path}')

        for local_index, environment_step in enumerate(plan_steps):
            environment_step = int(environment_step)
            subgoal = predicted_paths[local_index, -1]
            current = current_embeddings[local_index]
            encoded_current_mse = float(mse(frame_latents[environment_step], current))
            # The checkpoint runs in bfloat16, so encoding the same pixels in a
            # separately compiled batch can differ slightly from the online call.
            if encoded_current_mse > 1e-3:
                raise ValueError(
                    f'Trace current latent mismatch ({encoded_current_mse}) in '
                    f'{trace_path} at environment step {environment_step}.'
                )
            imagined = imagined_paths[local_index]
            blocks = environment_blocks[local_index]
            if len(imagined) != len(blocks):
                raise ValueError(f'Imagined path/action length mismatch: {trace_path}')
            normalized_blocks = (
                (blocks - action_mean[None, None]) / action_std[None, None]
            ).reshape(len(blocks), -1)
            event_index = len(event_rows)
            previous = np.concatenate((current[None], imagined[:-1]), axis=0)
            for block_index, (start_z, end_z, action) in enumerate(
                zip(previous, imagined, normalized_blocks)
            ):
                acid_current.append(start_z)
                acid_next.append(end_z)
                acid_actions.append(action)
                key = jax.random.fold_in(base_key, episode_index)
                key = jax.random.fold_in(key, environment_step)
                key = jax.random.fold_in(key, block_index)
                acid_keys.append(np.asarray(key, dtype=np.uint32))
                acid_event_indices.append(event_index)

            final_step = min(environment_step + args.real_horizon, len(frame_latents) - 1)
            future = frame_latents[environment_step + 1 : final_step + 1]
            start_mse = float(mse(current, subgoal))
            if len(future):
                future_mse = mse(future, subgoal)
                min_real_mse = float(future_mse.min())
                terminal_real_mse = float(future_mse[-1])
                relative_min_mse = min_real_mse / max(start_mse, 1e-12)
            else:
                min_real_mse = terminal_real_mse = relative_min_mse = float('nan')
            first_real_step = environment_step + args.transition_steps
            first_realization_mse = (
                float(mse(frame_latents[first_real_step], imagined[0]))
                if first_real_step < len(frame_latents)
                else float('nan')
            )
            imagined_subgoal_mse = float(np.min(mse(imagined, subgoal)))
            event_rows.append(
                {
                    'episode_index': episode_index,
                    'environment_step': environment_step,
                    'episode_success': success,
                    'trace_current_mse': encoded_current_mse,
                    'start_subgoal_mse': start_mse,
                    'min_real_subgoal_mse': min_real_mse,
                    'terminal_real_subgoal_mse': terminal_real_mse,
                    'relative_min_subgoal_mse': relative_min_mse,
                    'reach_at_0.50': relative_min_mse <= 0.50,
                    'reach_at_0.25': relative_min_mse <= 0.25,
                    'first_block_realization_mse': first_realization_mse,
                    'imagined_subgoal_mse': imagined_subgoal_mse,
                }
            )

    acid_current = jnp.asarray(np.stack(acid_current), dtype=jnp.float32)
    acid_next = jnp.asarray(np.stack(acid_next), dtype=jnp.float32)
    acid_keys = jnp.asarray(np.stack(acid_keys), dtype=jnp.uint32)
    acid_predictions = np.asarray(
        jax.device_get(inverse_one_step(acid_current, acid_next, acid_keys))
    )
    acid_actions = np.asarray(acid_actions, dtype=np.float32)
    block_errors = np.mean(np.square(acid_predictions - acid_actions), axis=-1)
    acid_event_indices = np.asarray(acid_event_indices, dtype=np.int32)
    for event_index, row in enumerate(event_rows):
        selected = block_errors[acid_event_indices == event_index]
        row['acid_error'] = float(selected.mean())
        row['acid_first_block_error'] = float(selected[0])

    columns = {
        key: np.asarray([row[key] for row in event_rows])
        for key in event_rows[0]
    }
    acid_error = columns['acid_error'].astype(np.float64)
    relative = columns['relative_min_subgoal_mse'].astype(np.float64)
    reach50 = columns['reach_at_0.50'].astype(bool)
    reach25 = columns['reach_at_0.25'].astype(bool)
    summary = {
        'task': args.task,
        'trace_dir': str(Path(args.trace_dir).resolve()),
        'lewm_checkpoint': str(Path(args.lewm_checkpoint).resolve()),
        'idm_checkpoint': str(Path(args.idm_checkpoint).resolve()),
        'idm_step': idm_step,
        'event_count': len(event_rows),
        'block_count': len(block_errors),
        'metrics': {
            'acid_error_mean': safe_mean(acid_error),
            'acid_error_std': safe_std(acid_error),
            'acid_first_block_error_mean': safe_mean(
                columns['acid_first_block_error']
            ),
            'real_min_subgoal_mse_mean': safe_mean(
                columns['min_real_subgoal_mse']
            ),
            'real_terminal_subgoal_mse_mean': safe_mean(
                columns['terminal_real_subgoal_mse']
            ),
            'relative_min_subgoal_mse_mean': safe_mean(relative),
            'reach_at_0.50': safe_mean(reach50),
            'reach_at_0.25': safe_mean(reach25),
            'first_block_realization_mse_mean': safe_mean(
                columns['first_block_realization_mse']
            ),
            'imagined_subgoal_mse_mean': safe_mean(
                columns['imagined_subgoal_mse']
            ),
            'trace_current_mse_max': float(
                np.max(columns['trace_current_mse'].astype(np.float64))
            ),
            'acid_vs_relative_distance_pearson': correlation(acid_error, relative),
            'acid_vs_relative_distance_spearman': correlation(
                acid_error, relative, rank=True
            ),
            'acid_predicts_reach_0.50_auc': binary_auc(reach50, -acid_error),
            'acid_predicts_reach_0.25_auc': binary_auc(reach25, -acid_error),
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    serializable_summary = strict_json(summary)
    output.write_text(json.dumps(serializable_summary, indent=2, sort_keys=True) + '\n')
    np.savez_compressed(output.with_suffix('.events.npz'), **columns)
    print(json.dumps(serializable_summary, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
