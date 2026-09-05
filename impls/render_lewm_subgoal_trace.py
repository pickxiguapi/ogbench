"""Render exact closed-loop H50 traces as Real / Imagination / Subgoal rows."""

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
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--decoder-batch-size', type=int, default=32)
    return parser.parse_args()


def outlined(array, color, width=4):
    image = Image.fromarray(array)
    ImageDraw.Draw(image).rectangle(
        (1, 1, image.width - 2, image.height - 2), outline=color, width=width
    )
    return image


def mse(left, right):
    return float(
        np.square(left.astype(np.float32) - right.astype(np.float32)).mean()
    )


def render_episode(path, frames, goal, events, goal_offset, waypoint_step, task, success):
    image_height, image_width = frames.shape[1:3]
    header_height = 122
    column_header_height = 34
    row_label_height = 26
    row_gap = 8
    row_height = row_label_height + image_height + row_gap
    canvas = Image.new(
        'RGB',
        (3 * image_width, header_height + column_header_height + len(events) * row_height),
        'white',
    )
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (10, 10),
        f'{task.upper()}  |  real H={goal_offset} closed-loop rollout  |  '
        f'{"success" if success else "failure"}',
        fill=(0, 0, 0),
    )
    draw.text(
        (10, 34),
        f'Each row starts a new plan at t; all three columns show its t+{waypoint_step} target.',
        fill=(65, 65, 65),
    )
    goal_thumb = Image.fromarray(goal).resize((72, 72), Image.Resampling.BILINEAR)
    goal_x = 3 * image_width - 82
    canvas.paste(goal_thumb, (goal_x, 6))
    draw.rectangle((goal_x, 6, goal_x + 71, 77), outline=(0, 145, 0), width=3)
    draw.text((goal_x - 5, 84), f'fixed goal t={goal_offset}', fill=(0, 110, 0))

    headers = (
        ('REAL FUTURE', (25, 25, 25)),
        ('LeWM IMAGINATION', (0, 80, 180)),
        ('PREDICTED SUBGOAL', (195, 20, 20)),
    )
    for column, (label, color) in enumerate(headers):
        x = column * image_width
        draw.rectangle(
            (x, header_height, x + image_width - 1, header_height + column_header_height - 1),
            fill=(244, 244, 244),
            outline=(215, 215, 215),
        )
        draw.text((x + 8, header_height + 10), label, fill=color)

    blank = Image.new('RGB', (image_width, image_height), (242, 242, 242))
    for row, event in enumerate(events):
        y = header_height + column_header_height + row * row_height
        plan_step = event['plan_step']
        target_step = event['target_step']
        if target_step < len(frames):
            real = outlined(frames[target_step], (35, 35, 35), width=3)
            real_note = f't={target_step}  (observed later)'
        else:
            real = blank.copy()
            ImageDraw.Draw(real).text((44, image_height // 2), 'trajectory ended', fill=(100, 100, 100))
            real_note = f't={target_step}  (not reached)'
        imagined = outlined(event['imagined_pixels'], (0, 90, 205))
        subgoal = outlined(event['subgoal_pixels'], (225, 25, 25))
        cells = (
            (real, real_note, (25, 25, 25)),
            (imagined, f'from real t={plan_step}', (0, 80, 180)),
            (subgoal, f'from real t={plan_step}', (195, 20, 20)),
        )
        for column, (cell, label, color) in enumerate(cells):
            x = column * image_width
            draw.text((x + 6, y + 7), label, fill=color)
            canvas.paste(cell, (x, y + row_label_height))

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
            required = {'frames', 'goal', 'plan_steps', 'predicted_paths', 'imagined_paths'}
            missing = required.difference(trace.files)
            if missing:
                raise ValueError(f'{trace_path} is missing trace arrays: {sorted(missing)}')
            frames = np.asarray(trace['frames'], dtype=np.uint8)
            goal = np.asarray(trace['goal'], dtype=np.uint8)
            plan_steps = np.asarray(trace['plan_steps'], dtype=np.int32)
            predicted_paths = np.asarray(trace['predicted_paths'], dtype=np.float32)
            imagined_paths = np.asarray(trace['imagined_paths'], dtype=np.float32)
            success = bool(trace['success'])
            episode = int(trace['episode'])
            start = int(trace['start'])

        keep = plan_steps + args.waypoint_step <= args.goal_offset
        kept_steps = plan_steps[keep]
        subgoal_latents = predicted_paths[keep, -1]
        imagined_latents = imagined_paths[keep, -1]
        all_latents = np.concatenate([imagined_latents, subgoal_latents], axis=0)
        decoded = (
            decode_latents(decoder, all_latents, args.decoder_batch_size, args.device)
            if len(all_latents)
            else np.empty((0, 224, 224, 3), dtype=np.uint8)
        )
        imagined_decoded = decoded[: len(imagined_latents)]
        subgoal_decoded = decoded[len(imagined_latents) :]

        events = []
        for plan_step, imagined_pixels, subgoal_pixels in zip(
            kept_steps.tolist(), imagined_decoded, subgoal_decoded
        ):
            target_step = int(plan_step + args.waypoint_step)
            real_pixels = frames[target_step] if target_step < len(frames) else None
            events.append(
                {
                    'plan_step': int(plan_step),
                    'target_step': target_step,
                    'imagined_pixels': imagined_pixels,
                    'subgoal_pixels': subgoal_pixels,
                    'lewm_imagination_vs_actual_mse_0_255': (
                        mse(imagined_pixels, real_pixels) if real_pixels is not None else None
                    ),
                    'decoded_subgoal_vs_actual_mse_0_255': (
                        mse(subgoal_pixels, real_pixels) if real_pixels is not None else None
                    ),
                    'lewm_imagination_vs_subgoal_mse_0_255': mse(
                        imagined_pixels, subgoal_pixels
                    ),
                }
            )

        figure_name = f'{trace_path.stem}_real_imagination_subgoal_h{args.goal_offset}.png'
        render_episode(
            output_dir / figure_name,
            frames,
            goal,
            events,
            args.goal_offset,
            args.waypoint_step,
            args.task,
            success,
        )
        summaries.append(
            {
                'trace': trace_path.name,
                'figure': figure_name,
                'episode': episode,
                'start': start,
                'success': success,
                'executed_steps': len(frames) - 1,
                'events': [
                    {key: value for key, value in event.items() if not key.endswith('_pixels')}
                    for event in events
                ],
            }
        )

    manifest = {
        'task': args.task,
        'protocol': (
            'exact closed-loop H50 policy trace; each row aligns the real future frame, '
            'the LeWM terminal imagination under the CEM-selected two-block action plan, '
            'and the general uniform-future generator K10 subgoal decoded with one ConvDecoder'
        ),
        'columns': ['real future', 'LeWM imagination', 'predicted subgoal'],
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
