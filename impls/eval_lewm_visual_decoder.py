"""Evaluate a frozen LeWM image decoder and render predicted latent subgoals.

The protocol uses held-out episodes only.  For each pair, the maxgoal25
LatentPathFlow generator receives three cached LeWM history latents and the
true t+25 goal latent, and predicts the t+5/t+10 waypoint latents.  The same
frozen decoder renders both target and predicted latents.
"""

from __future__ import annotations

import argparse
import io
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image, ImageDraw

from latent_subgoal import (
    LATENT_PATH_FLOW_ARCHITECTURE,
    latent_path_waypoint_steps,
    load_latent_subgoal_checkpoint,
    sample_conditional_path_flow_candidates,
    select_latent_path_medoid,
)
from utils.latent_subgoal_dataset import (
    build_history_indices,
    build_valid_transitions,
    load_latent_cache,
    split_episodes,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=('pusht', 'cube', 'reacher', 'tworoom'), required=True)
    parser.add_argument('--latent-hdf5', required=True)
    parser.add_argument('--lance-path', required=True)
    parser.add_argument('--decoder-checkpoint', required=True)
    parser.add_argument('--subgoal-checkpoint', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--num-pairs', type=int, default=256)
    parser.add_argument('--num-samples', type=int, default=8)
    parser.add_argument('--generator-batch-size', type=int, default=32)
    parser.add_argument('--decoder-batch-size', type=int, default=32)
    parser.add_argument('--goal-offset', type=int, default=25)
    parser.add_argument('--split-seed', type=int, default=0)
    parser.add_argument('--sampling-seed', type=int, default=42)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--decode-workers', type=int, default=12)
    parser.add_argument('--phase', choices=('all', 'predict', 'render'), default='all')
    parser.add_argument('--prediction-file', default=None)
    return parser.parse_args()


class LancePixels:
    def __init__(self, path, workers):
        import lancedb
        from lancedb.permutation import Permutation

        path = Path(path)
        table = lancedb.connect(str(path.parent)).open_table(path.stem)
        self.rows = Permutation.identity(table).select_columns(['pixels']).with_format('arrow')
        self.executor = ThreadPoolExecutor(max_workers=max(1, workers))

    @staticmethod
    def decode(blob):
        with Image.open(io.BytesIO(blob)) as image:
            return np.asarray(image.convert('RGB'), dtype=np.uint8).copy()

    def fetch(self, indices):
        indices = np.asarray(indices, dtype=np.int64)
        unique, inverse = np.unique(indices, return_inverse=True)
        batch = self.rows.__getitems__(unique.tolist())
        blobs = batch.column(batch.schema.get_field_index('pixels')).to_pylist()
        decoded = np.stack(list(self.executor.map(self.decode, blobs)))
        return decoded[inverse]

    def close(self):
        self.executor.shutdown(wait=True)


def predict_paths(model, params, config, histories, goals, args):
    num_steps = int(config['flow_sampling_steps'])
    solver = str(config['flow_solver'])
    predict = jax.jit(
        lambda current, goal, rng: select_latent_path_medoid(
            sample_conditional_path_flow_candidates(
                model,
                params,
                current,
                goal,
                rng,
                num_samples=args.num_samples,
                num_steps=num_steps,
                solver=solver,
            )
        )
    )
    outputs = []
    root_rng = jax.random.PRNGKey(args.sampling_seed)
    for start in range(0, len(histories), args.generator_batch_size):
        stop = min(start + args.generator_batch_size, len(histories))
        rng = jax.random.fold_in(root_rng, start)
        output = predict(
            jnp.asarray(histories[start:stop]), jnp.asarray(goals[start:stop]), rng
        )
        outputs.append(np.asarray(output, dtype=np.float32))
    return np.concatenate(outputs)


def decode_latents(decoder, latents, batch_size, device):
    import torch

    outputs = []
    with torch.no_grad():
        for start in range(0, len(latents), batch_size):
            z = torch.from_numpy(latents[start : start + batch_size]).to(device)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                prediction = decoder(z)
            outputs.append(prediction.float().clamp(-1, 1).cpu())
    return torch.cat(outputs).add(1).mul(127.5).round().byte().permute(0, 2, 3, 1).numpy()


def cosine_similarity(prediction, target):
    numerator = np.sum(prediction * target, axis=-1)
    denominator = np.linalg.norm(prediction, axis=-1) * np.linalg.norm(target, axis=-1)
    return numerator / np.maximum(denominator, 1e-12)


def pixel_metrics(prediction, target, current):
    import torch
    import torch.nn.functional as F

    squared = np.square(prediction.astype(np.float32) - target.astype(np.float32))
    mse = squared.mean(axis=(1, 2, 3))
    motion = np.max(np.abs(target.astype(np.int16) - current.astype(np.int16)), axis=-1) > 12
    motion = torch.from_numpy(motion[:, None].astype(np.float32))
    motion = F.max_pool2d(motion, kernel_size=9, stride=1, padding=4)
    motion = motion[:, 0].numpy() > 0
    motion_mse = np.asarray(
        [sample[mask].mean() if np.any(mask) else np.nan for sample, mask in zip(squared.mean(-1), motion)]
    )
    return mse, motion_mse, motion.mean(axis=(1, 2))


def add_label(image, text, color=(0, 0, 0)):
    image = Image.fromarray(image)
    canvas = Image.new('RGB', (image.width, image.height + 24), 'white')
    canvas.paste(image, (0, 24))
    ImageDraw.Draw(canvas).text((5, 5), text, fill=color)
    return canvas


def make_sheet(path, indices, originals, decoded_true, decoded_pred, latent_mse):
    columns = ('t', 'GT t+5', 'GT t+10', 'goal t+25')
    height, width = originals.shape[2], originals.shape[3]
    row_height = height + 24
    separator = 6
    canvas = Image.new(
        'RGB',
        (4 * width, len(indices) * (3 * row_height + separator)),
        'white',
    )
    for group, index in enumerate(indices):
        top = group * (3 * row_height + separator)
        for column, label in enumerate(columns):
            canvas.paste(add_label(originals[index, column], f'RGB {label}'), (column * width, top))
            canvas.paste(
                add_label(decoded_true[index, column], f'Dec(E) {label}'),
                (column * width, top + row_height),
            )
        predicted_cells = [None, decoded_pred[index, 0], decoded_pred[index, 1], None]
        for column, cell in enumerate(predicted_cells):
            if cell is None:
                cell = np.full((height, width, 3), 255, dtype=np.uint8)
                label = ''
            else:
                waypoint = 5 if column == 1 else 10
                label = f'Pred subgoal t+{waypoint}  zMSE={latent_mse[index, column - 1]:.3f}'
            canvas.paste(add_label(cell, label, color=(200, 0, 0)), (column * width, top + 2 * row_height))
    canvas.save(path)


def prediction_file(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return Path(args.prediction_file) if args.prediction_file else output_dir / 'predictions.npz'


def predict_and_save(args):
    cache = load_latent_cache(args.latent_hdf5)
    model, params, config, checkpoint_step = load_latent_subgoal_checkpoint(
        args.subgoal_checkpoint
    )
    if config['architecture'] != LATENT_PATH_FLOW_ARCHITECTURE:
        raise ValueError('This evaluator requires a LatentPathFlow checkpoint.')
    if int(config.get('max_goal_steps', -1)) != args.goal_offset:
        raise ValueError('Generator max_goal_steps must match goal_offset.')
    history_size = int(config['history_size'])
    waypoint_steps = latent_path_waypoint_steps(
        int(config['subgoal_steps']), int(config['action_block'])
    )
    if waypoint_steps != (5, 10):
        raise ValueError(f'Expected K5/K10 waypoints, got {waypoint_steps}.')

    _, validation_episodes = split_episodes(
        len(cache.episode_offsets), 0.95, args.split_seed
    )
    valid_t, _ = build_valid_transitions(
        cache.episode_offsets,
        cache.episode_lengths,
        validation_episodes,
        min_future_steps=args.goal_offset,
    )
    rng = np.random.default_rng(args.sampling_seed)
    replace = len(valid_t) < args.num_pairs
    current_rows = rng.choice(valid_t, size=args.num_pairs, replace=replace).astype(np.int64)
    history_rows = build_history_indices(
        current_rows, cache.episode_offsets, history_size
    )
    goal_rows = current_rows + args.goal_offset
    target_rows = np.stack([current_rows + step for step in waypoint_steps], axis=1)

    predictions = predict_paths(
        model,
        params,
        config,
        cache.z[history_rows],
        cache.z[goal_rows],
        args,
    )
    targets = cache.z[target_rows]
    latent_mse = np.mean(np.square(predictions - targets), axis=-1)
    latent_cosine = cosine_similarity(predictions, targets)

    output = prediction_file(args)
    temporary = output.with_suffix(output.suffix + '.tmp')
    with temporary.open('wb') as file:
        np.savez_compressed(
            file,
            current_rows=current_rows,
            history_rows=history_rows,
            goal_rows=goal_rows,
            target_rows=target_rows,
            predictions=predictions,
            latent_mse=latent_mse,
            latent_cosine=latent_cosine,
            generator_checkpoint_step=np.asarray(checkpoint_step),
            lewm_checkpoint_sha256=np.asarray(config['lewm_checkpoint_sha256']),
        )
    temporary.replace(output)
    print(f'Saved {len(current_rows)} latent predictions to {output}', flush=True)


def render_predictions(args):
    from lewm_visual_decoder import load_decoder

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache = load_latent_cache(args.latent_hdf5)
    with np.load(prediction_file(args), allow_pickle=False) as saved:
        current_rows = np.asarray(saved['current_rows'], dtype=np.int64)
        goal_rows = np.asarray(saved['goal_rows'], dtype=np.int64)
        predictions = np.asarray(saved['predictions'], dtype=np.float32)
        latent_mse = np.asarray(saved['latent_mse'], dtype=np.float32)
        latent_cosine = np.asarray(saved['latent_cosine'], dtype=np.float32)
        checkpoint_step = int(saved['generator_checkpoint_step'])
        generator_sha = str(saved['lewm_checkpoint_sha256'].item())
    num_pairs = len(current_rows)

    decoder, decoder_payload = load_decoder(args.decoder_checkpoint, args.device)
    decoder_sha = decoder_payload['manifest']['lewm_checkpoint_sha256']
    if decoder_sha != generator_sha or decoder_sha != cache.metadata['checkpoint_sha256']:
        raise ValueError('Decoder, generator, and latent cache bind different LeWM checkpoints.')
    true_rows = np.stack(
        [current_rows, current_rows + 5, current_rows + 10, goal_rows], axis=1
    )
    decoded_true = decode_latents(
        decoder,
        cache.z[true_rows].reshape(-1, cache.z.shape[1]),
        args.decoder_batch_size,
        args.device,
    ).reshape(num_pairs, 4, 224, 224, 3)
    decoded_pred = decode_latents(
        decoder,
        predictions.reshape(-1, cache.z.shape[1]),
        args.decoder_batch_size,
        args.device,
    ).reshape(num_pairs, 2, 224, 224, 3)

    pixels = LancePixels(args.lance_path, args.decode_workers)
    try:
        originals = pixels.fetch(true_rows.reshape(-1)).reshape(
            num_pairs, 4, 224, 224, 3
        )
    finally:
        pixels.close()

    reconstruction_mse, reconstruction_motion_mse, motion_fraction = pixel_metrics(
        decoded_true[:, 1:3].reshape(-1, 224, 224, 3),
        originals[:, 1:3].reshape(-1, 224, 224, 3),
        np.repeat(originals[:, :1], 2, axis=1).reshape(-1, 224, 224, 3),
    )
    prediction_mse, prediction_motion_mse, _ = pixel_metrics(
        decoded_pred.reshape(-1, 224, 224, 3),
        originals[:, 1:3].reshape(-1, 224, 224, 3),
        np.repeat(originals[:, :1], 2, axis=1).reshape(-1, 224, 224, 3),
    )

    score = latent_mse.mean(axis=1)
    order = np.argsort(score)
    best = order[: min(6, len(order))]
    quantiles = np.linspace(0.1, 0.9, min(6, len(order)))
    representative = order[np.round(quantiles * (len(order) - 1)).astype(np.int64)]
    make_sheet(
        output_dir / 'best_cases.png', best, originals, decoded_true, decoded_pred, latent_mse
    )
    make_sheet(
        output_dir / 'representative_quantiles.png',
        representative,
        originals,
        decoded_true,
        decoded_pred,
        latent_mse,
    )

    metrics = {
        'task': args.task,
        'protocol': 'held-out episodes; history3; exact t+25 goal; medoid of sampled K5/K10 paths',
        'num_pairs': num_pairs,
        'num_samples': args.num_samples,
        'generator_checkpoint_step': checkpoint_step,
        'decoder_checkpoint_epoch': int(decoder_payload['epoch']),
        'latent': {
            'waypoint_5_mse': float(latent_mse[:, 0].mean()),
            'waypoint_10_mse': float(latent_mse[:, 1].mean()),
            'waypoint_5_cosine': float(latent_cosine[:, 0].mean()),
            'waypoint_10_cosine': float(latent_cosine[:, 1].mean()),
        },
        'pixels': {
            'decoded_true_mse_0_255': float(np.mean(reconstruction_mse)),
            'decoded_prediction_mse_0_255': float(np.mean(prediction_mse)),
            'decoded_true_motion_mse_0_255': float(np.nanmean(reconstruction_motion_mse)),
            'decoded_prediction_motion_mse_0_255': float(np.nanmean(prediction_motion_mse)),
            'motion_mask_fraction': float(np.mean(motion_fraction)),
        },
        'selection': {
            'best_case_indices': best.tolist(),
            'representative_quantile_indices': representative.tolist(),
        },
    }
    (output_dir / 'metrics.json').write_text(json.dumps(metrics, indent=2, sort_keys=True))
    print(json.dumps(metrics, indent=2, sort_keys=True), flush=True)


def main():
    args = parse_args()
    if args.phase in ('all', 'predict'):
        predict_and_save(args)
    if args.phase in ('all', 'render'):
        render_predictions(args)


if __name__ == '__main__':
    main()
