"""Rank and render continuous Real / LeWM imagination / subgoal figure candidates."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


HEADER_HEIGHT = 122
COLUMN_HEADER_HEIGHT = 34
ROW_LABEL_HEIGHT = 26
SOURCE_CELL = 224
SOURCE_ROW_HEIGHT = 26 + SOURCE_CELL + 8


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-root', required=True)
    parser.add_argument('--output-root', required=True)
    parser.add_argument('--tasks', nargs='+', default=['cube', 'pusht'])
    parser.add_argument('--window', type=int, default=6)
    parser.add_argument('--topk', type=int, default=12)
    parser.add_argument('--from-zero', action='store_true')
    parser.add_argument('--display-step', type=int, default=10)
    parser.add_argument('--pusht-min-displacement', type=float, default=30.0)
    return parser.parse_args()


def load_font(size, bold=False):
    candidates = [
        '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf' if bold
        else '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
        '/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf' if bold
        else '/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf',
    ]
    for path in candidates:
        if Path(path).is_file():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def crop_cell(composite, row, column):
    top = HEADER_HEIGHT + COLUMN_HEADER_HEIGHT + row * SOURCE_ROW_HEIGHT + ROW_LABEL_HEIGHT
    left = column * SOURCE_CELL
    return composite.crop((left, top, left + SOURCE_CELL, top + SOURCE_CELL))


def outlined(array, color, width=4):
    image = Image.fromarray(array)
    ImageDraw.Draw(image).rectangle(
        (1, 1, image.width - 2, image.height - 2), outline=color, width=width
    )
    return image


def blue_centroid(frame):
    """Return the PushT blue object's image-space centroid."""
    pixels = frame.astype(np.int16)
    mask = (
        (pixels[..., 2] > 130)
        & (pixels[..., 2] > pixels[..., 0] + 35)
        & (pixels[..., 2] > pixels[..., 1] + 15)
        & (pixels[..., 0] < 170)
    )
    rows, columns = np.nonzero(mask)
    if len(columns) < 5:
        return None
    return np.asarray([columns.mean(), rows.mean()], dtype=np.float64)


def push_t_motion(frames, display_steps):
    centroids = [blue_centroid(frames[step]) for step in display_steps]
    if any(centroid is None for centroid in centroids):
        return None, None
    centroids = np.stack(centroids)
    displacement = float(np.linalg.norm(centroids[-1] - centroids[0]))
    path_length = float(np.linalg.norm(np.diff(centroids, axis=0), axis=1).sum())
    return displacement, path_length


def best_window(episode, window):
    events = episode['events']
    candidates = []
    for start in range(max(len(events) - window + 1, 0)):
        selected = events[start : start + window]
        if any(event['lewm_imagination_vs_actual_mse_0_255'] is None for event in selected):
            continue
        if any(event['decoded_subgoal_vs_actual_mse_0_255'] is None for event in selected):
            continue
        dual = np.asarray(
            [
                max(
                    event['lewm_imagination_vs_actual_mse_0_255'],
                    event['decoded_subgoal_vs_actual_mse_0_255'],
                )
                for event in selected
            ],
            dtype=np.float64,
        )
        candidates.append(
            {
                'start_row': start,
                'end_row': start + window,
                'mean_worst_branch_mse': float(dual.mean()),
                'max_worst_branch_mse': float(dual.max()),
                'mean_imagination_mse': float(
                    np.mean([event['lewm_imagination_vs_actual_mse_0_255'] for event in selected])
                ),
                'mean_subgoal_mse': float(
                    np.mean([event['decoded_subgoal_vs_actual_mse_0_255'] for event in selected])
                ),
                'plan_steps': [int(event['plan_step']) for event in selected],
                'target_steps': [int(event['target_step']) for event in selected],
            }
        )
    return min(candidates, key=lambda item: (item['mean_worst_branch_mse'], item['max_worst_branch_mse'])) if candidates else None


def from_zero_candidate(episode, trace_path, goal_offset, waypoint_step, display_step, task):
    display_steps = list(range(0, goal_offset + 1, display_step))
    target_steps = list(range(waypoint_step, goal_offset + 1, display_step))
    events_by_target = {
        int(event['target_step']): (row, event)
        for row, event in enumerate(episode['events'])
    }
    if any(step not in events_by_target for step in target_steps):
        return None
    selected = [events_by_target[step][1] for step in target_steps]
    if any(event['lewm_imagination_vs_actual_mse_0_255'] is None for event in selected):
        return None
    if any(event['decoded_subgoal_vs_actual_mse_0_255'] is None for event in selected):
        return None
    with np.load(trace_path, allow_pickle=False) as trace:
        frames = np.asarray(trace['frames'], dtype=np.uint8)
    if len(frames) <= goal_offset:
        return None
    dual = np.asarray(
        [
            max(
                event['lewm_imagination_vs_actual_mse_0_255'],
                event['decoded_subgoal_vs_actual_mse_0_255'],
            )
            for event in selected
        ],
        dtype=np.float64,
    )
    displacement, path_length = (None, None)
    if task == 'pusht':
        displacement, path_length = push_t_motion(frames, display_steps)
    return {
        'mean_worst_branch_mse': float(dual.mean()),
        'max_worst_branch_mse': float(dual.max()),
        'mean_imagination_mse': float(
            np.mean([event['lewm_imagination_vs_actual_mse_0_255'] for event in selected])
        ),
        'mean_subgoal_mse': float(
            np.mean([event['decoded_subgoal_vs_actual_mse_0_255'] for event in selected])
        ),
        'display_steps': display_steps,
        'target_steps': target_steps,
        'plan_steps': [int(event['plan_step']) for event in selected],
        'source_rows': [int(events_by_target[step][0]) for step in target_steps],
        'pusht_blue_displacement_px': displacement,
        'pusht_blue_path_length_px': path_length,
    }


def render_candidate(source, candidate, output, task, rank):
    composite = Image.open(source).convert('RGB')
    window = candidate['end_row'] - candidate['start_row']
    label_width = 230
    cell = SOURCE_CELL
    top_height = 66
    bottom_height = 52
    canvas = Image.new('RGB', (label_width + window * cell, top_height + 3 * cell + bottom_height), 'white')
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(25, bold=True)
    label_font = load_font(22, bold=True)
    step_font = load_font(20)
    draw.text((10, 13), f'{task.upper()} candidate #{rank}', fill='black', font=title_font)
    labels = [('Real', 'black'), ('LeWM imagined', (20, 95, 205)), ('Predicted subgoal', (215, 30, 40))]
    for output_row, (label, color) in enumerate(labels):
        y = top_height + output_row * cell
        draw.text((10, y + cell // 2 - 16), label, fill=color, font=label_font)
        for output_column, source_row in enumerate(range(candidate['start_row'], candidate['end_row'])):
            frame = crop_cell(composite, source_row, output_row)
            x = label_width + output_column * cell
            canvas.paste(frame, (x, y))
    for column, (plan_step, target_step) in enumerate(zip(candidate['plan_steps'], candidate['target_steps'])):
        x = label_width + column * cell
        text = f't={target_step}\n(from {plan_step})'
        draw.multiline_text(
            (x + cell // 2, top_height + 3 * cell + 5),
            text,
            fill='black',
            font=step_font,
            spacing=1,
            anchor='ma',
            align='center',
        )
    canvas.save(output, quality=95)


def blank_prediction_cell(color):
    cell = Image.new('RGB', (SOURCE_CELL, SOURCE_CELL), (247, 247, 247))
    draw = ImageDraw.Draw(cell)
    draw.rectangle((1, 1, SOURCE_CELL - 2, SOURCE_CELL - 2), outline=color, width=4)
    draw.multiline_text(
        (SOURCE_CELL // 2, SOURCE_CELL // 2),
        'context only\n(no prediction)',
        fill=(120, 120, 120),
        font=load_font(16),
        spacing=3,
        anchor='mm',
        align='center',
    )
    return cell


def render_from_zero_candidate(source, trace_path, candidate, output, task, rank):
    composite = Image.open(source).convert('RGB')
    with np.load(trace_path, allow_pickle=False) as trace:
        frames = np.asarray(trace['frames'], dtype=np.uint8)
    display_steps = candidate['display_steps']
    label_width = 230
    cell = SOURCE_CELL
    top_height = 66
    bottom_height = 52
    canvas = Image.new(
        'RGB',
        (label_width + len(display_steps) * cell, top_height + 3 * cell + bottom_height),
        'white',
    )
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(25, bold=True)
    label_font = load_font(22, bold=True)
    step_font = load_font(18)
    draw.text((10, 13), f'{task.upper()} candidate #{rank}', fill='black', font=title_font)
    labels = [('Real', 'black'), ('LeWM imagined', (20, 95, 205)), ('Predicted subgoal', (215, 30, 40))]
    source_rows = dict(zip(candidate['target_steps'], candidate['source_rows']))
    red_blank = blank_prediction_cell((225, 25, 25))
    for output_row, (label, color) in enumerate(labels):
        y = top_height + output_row * cell
        draw.text((10, y + cell // 2 - 16), label, fill=color, font=label_font)
        for output_column, step in enumerate(display_steps):
            x = label_width + output_column * cell
            if output_row == 0:
                frame = outlined(frames[step], (35, 35, 35), width=3)
            elif step in source_rows:
                frame = crop_cell(composite, source_rows[step], output_row)
            elif output_row == 1:
                # The rollout starts from observed context; do not label it as a forecast.
                frame = outlined(frames[step], (0, 90, 205), width=4)
            else:
                frame = red_blank
            canvas.paste(frame, (x, y))
    for column, step in enumerate(display_steps):
        x = label_width + column * cell
        if step < candidate['target_steps'][0]:
            note = 'context'
        else:
            plan_step = candidate['plan_steps'][candidate['target_steps'].index(step)]
            note = f'from {plan_step}'
        draw.multiline_text(
            (x + cell // 2, top_height + 3 * cell + 5),
            f't={step}\n({note})',
            fill='black',
            font=step_font,
            spacing=1,
            anchor='ma',
            align='center',
        )
    canvas.save(output, quality=95)


def rank_task(input_root, output_root, task, window, topk):
    figure_dir = input_root / task / 'figures'
    manifest_path = figure_dir / 'manifest.json'
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    candidates = []
    for episode_index, episode in enumerate(manifest['episodes']):
        candidate = best_window(episode, window)
        if candidate is None:
            continue
        candidate.update(
            {
                'episode_index': episode_index,
                'episode': int(episode['episode']),
                'start': int(episode['start']),
                'success': bool(episode['success']),
                'executed_steps': int(episode['executed_steps']),
                'source_figure': episode['figure'],
            }
        )
        candidates.append(candidate)
    candidates.sort(key=lambda item: (item['mean_worst_branch_mse'], item['max_worst_branch_mse']))
    selected = candidates[:topk]
    task_output = output_root / task
    task_output.mkdir(parents=True, exist_ok=True)
    for rank, candidate in enumerate(selected, 1):
        name = f'rank_{rank:02d}_episode_{candidate["episode_index"]:03d}_plans_{candidate["plan_steps"][0]}_{candidate["plan_steps"][-1]}.png'
        render_candidate(
            figure_dir / candidate['source_figure'],
            candidate,
            task_output / name,
            task,
            rank,
        )
        candidate['curated_figure'] = name
        candidate['rank'] = rank
    (task_output / 'ranking.json').write_text(json.dumps(selected, indent=2))
    fields = [
        'rank', 'episode_index', 'episode', 'start', 'success', 'executed_steps',
        'mean_worst_branch_mse', 'max_worst_branch_mse',
        'mean_imagination_mse', 'mean_subgoal_mse', 'plan_steps', 'target_steps',
        'source_figure', 'curated_figure',
    ]
    with (task_output / 'ranking.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for candidate in selected:
            writer.writerow({key: candidate[key] for key in fields})
    return selected


def rank_task_from_zero(
    input_root,
    output_root,
    task,
    topk,
    display_step,
    pusht_min_displacement,
):
    figure_dir = input_root / task / 'figures'
    trace_dir = input_root / task / 'trace'
    manifest_path = figure_dir / 'manifest.json'
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    candidates = []
    for episode_index, episode in enumerate(manifest['episodes']):
        trace_path = trace_dir / episode['trace']
        if not trace_path.is_file():
            raise FileNotFoundError(trace_path)
        candidate = from_zero_candidate(
            episode,
            trace_path,
            int(manifest['goal_offset']),
            int(manifest['waypoint_step']),
            display_step,
            task,
        )
        if candidate is None:
            continue
        if task == 'pusht' and (
            candidate['pusht_blue_displacement_px'] is None
            or candidate['pusht_blue_displacement_px'] < pusht_min_displacement
        ):
            continue
        candidate.update(
            {
                'episode_index': episode_index,
                'episode': int(episode['episode']),
                'start': int(episode['start']),
                'success': bool(episode['success']),
                'executed_steps': int(episode['executed_steps']),
                'source_figure': episode['figure'],
                'source_trace': episode['trace'],
            }
        )
        candidates.append(candidate)
    candidates.sort(key=lambda item: (item['mean_worst_branch_mse'], item['max_worst_branch_mse']))
    selected = candidates[:topk]
    task_output = output_root / task
    task_output.mkdir(parents=True, exist_ok=True)
    for rank, candidate in enumerate(selected, 1):
        name = f'rank_{rank:02d}_episode_{candidate["episode_index"]:03d}_t0_50.png'
        render_from_zero_candidate(
            figure_dir / candidate['source_figure'],
            trace_dir / candidate['source_trace'],
            candidate,
            task_output / name,
            task,
            rank,
        )
        candidate['curated_figure'] = name
        candidate['rank'] = rank
    (task_output / 'ranking.json').write_text(json.dumps(selected, indent=2))
    fields = [
        'rank', 'episode_index', 'episode', 'start', 'success', 'executed_steps',
        'mean_worst_branch_mse', 'max_worst_branch_mse',
        'mean_imagination_mse', 'mean_subgoal_mse',
        'pusht_blue_displacement_px', 'pusht_blue_path_length_px',
        'display_steps', 'plan_steps', 'target_steps',
        'source_figure', 'source_trace', 'curated_figure',
    ]
    with (task_output / 'ranking.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for candidate in selected:
            writer.writerow({key: candidate[key] for key in fields})
    return selected


def main():
    args = parse_args()
    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    if args.from_zero:
        summary = {
            task: rank_task_from_zero(
                input_root,
                output_root,
                task,
                args.topk,
                args.display_step,
                args.pusht_min_displacement,
            )
            for task in args.tasks
        }
    else:
        summary = {
            task: rank_task(input_root, output_root, task, args.window, args.topk)
            for task in args.tasks
        }
    (output_root / 'ranking_summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps({task: len(items) for task, items in summary.items()}, indent=2))


if __name__ == '__main__':
    main()
