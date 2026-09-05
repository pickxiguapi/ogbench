"""Render exact closed-loop H50 subgoal traces captured by the evaluator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from eval_lewm_visual_decoder import decode_latents
from lewm_visual_decoder import load_decoder


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', required=True)
    parser.add_argument('--trace-dir', required=True)
    parser.add_argument('--decoder-checkpoint', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--goal-offset', type=int, default=50)
    parser.add_argument('--waypoint-step', type=int, default=10)
    parser.add_argument('--display-stride', type=int, default=5)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--decoder-batch-size', type=int, default=32)
    return parser.parse_args()


def outlined(array, color, width=4):
    image = Image.fromarray(array)
    draw = ImageDraw.Draw(image)
    draw.rectangle((1, 1, image.width - 2, image.height - 2), outline=color, width=width)
    return np.asarray(image)


def labeled(array, text, color=(0, 0, 0)):
    image = Image.fromarray(array)
    canvas = Image.new('RGB', (image.width, image.height + 24), 'white')
    canvas.paste(image, (0, 24))
    ImageDraw.Draw(canvas).text((5, 5), text, fill=color)
    return canvas


def render_episode(path, frames, goal, predictions, goal_offset, stride, waypoint_step):
    steps = list(range(0, goal_offset + 1, stride))
    height, width = frames.shape[1:3]
    row_height = height + 24
    blank = np.full((height, width, 3), 245, dtype=np.uint8)
    canvas = Image.new('RGB', ((len(steps) + 1) * width, 2 * row_height), 'white')

    for column, step in enumerate(steps):
        if step < len(frames):
            actual = frames[step]
            actual_label = f'Actual t={step}'
        else:
            actual = blank
            actual_label = f't={step} not reached'
        canvas.paste(labeled(actual, actual_label), (column * width, 0))

        event = predictions.get(step)
        if event is None:
            predicted = blank
            predicted_label = 'no K10 target' if step < waypoint_step else 'not available'
        else:
            predicted = outlined(event['pixels'], (230, 30, 30))
            predicted_label = f'Pred SG@{step} from t={event["plan_step"]}'
        canvas.paste(
            labeled(predicted, predicted_label, color=(190, 0, 0)),
            (column * width, row_height),
        )

    goal_column = len(steps)
    canvas.paste(
        labeled(outlined(goal, (0, 150, 0)), f'Fixed goal H={goal_offset}', color=(0, 110, 0)),
        (goal_column * width, 0),
    )
    canvas.paste(labeled(blank, 'same goal at every replan'), (goal_column * width, row_height))
    canvas.save(path)


def main():
    args = parse_args()
    trace_paths = sorted(Path(args.trace_dir).glob('episode_*.npz'))
    if not trace_paths:
        raise FileNotFoundError(f'No trace files found in {args.trace_dir}.')

    decoder, decoder_payload = load_decoder(args.decoder_checkpoint, args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []

    for trace_path in trace_paths:
        with np.load(trace_path, allow_pickle=False) as trace:
            frames = np.asarray(trace['frames'], dtype=np.uint8)
            goal = np.asarray(trace['goal'], dtype=np.uint8)
            plan_steps = np.asarray(trace['plan_steps'], dtype=np.int32)
            predicted_paths = np.asarray(trace['predicted_paths'], dtype=np.float32)
            success = bool(trace['success'])
            episode = int(trace['episode'])
            start = int(trace['start'])

        keep = plan_steps + args.waypoint_step <= args.goal_offset
        kept_steps = plan_steps[keep]
        kept_paths = predicted_paths[keep]
        terminal_latents = kept_paths[:, -1]
        decoded = (
            decode_latents(
                decoder,
                terminal_latents,
                args.decoder_batch_size,
                args.device,
            )
            if len(terminal_latents)
            else np.empty((0, 224, 224, 3), dtype=np.uint8)
        )

        predictions = {}
        event_metrics = []
        for plan_step, pixels in zip(kept_steps.tolist(), decoded):
            target_step = int(plan_step + args.waypoint_step)
            predictions[target_step] = {'plan_step': int(plan_step), 'pixels': pixels}
            if target_step < len(frames):
                mse = float(
                    np.square(
                        pixels.astype(np.float32) - frames[target_step].astype(np.float32)
                    ).mean()
                )
            else:
                mse = None
            event_metrics.append(
                {
                    'plan_step': int(plan_step),
                    'target_step': target_step,
                    'decoded_subgoal_vs_actual_mse_0_255': mse,
                }
            )

        figure_name = f'{trace_path.stem}_closed_loop_h{args.goal_offset}.png'
        render_episode(
            output_dir / figure_name,
            frames,
            goal,
            predictions,
            args.goal_offset,
            args.display_stride,
            args.waypoint_step,
        )
        finite_mse = [
            event['decoded_subgoal_vs_actual_mse_0_255']
            for event in event_metrics
            if event['decoded_subgoal_vs_actual_mse_0_255'] is not None
        ]
        summaries.append(
            {
                'trace': trace_path.name,
                'figure': figure_name,
                'episode': episode,
                'start': start,
                'success': success,
                'executed_steps': len(frames) - 1,
                'mean_decoded_subgoal_vs_actual_mse_0_255': (
                    float(np.mean(finite_mse)) if finite_mse else None
                ),
                'events': event_metrics,
            }
        )

    manifest = {
        'task': args.task,
        'protocol': (
            'exact closed-loop policy trace; fixed dataset goal at t0+H; '
            'general uniform-future generator refreshed every 5 executed steps; '
            'K10 target decoded and aligned with the corresponding future time'
        ),
        'goal_offset': args.goal_offset,
        'waypoint_step': args.waypoint_step,
        'decoder_checkpoint': str(Path(args.decoder_checkpoint).resolve()),
        'decoder_checkpoint_epoch': int(decoder_payload['epoch']),
        'episodes': summaries,
    }
    (output_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == '__main__':
    main()
