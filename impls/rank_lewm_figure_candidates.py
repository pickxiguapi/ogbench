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


def render_candidate(source, candidate, output, task, rank):
    composite = Image.open(source).convert('RGB')
    window = candidate['end_row'] - candidate['start_row']
    label_width = 150
    cell = SOURCE_CELL
    top_height = 66
    bottom_height = 52
    canvas = Image.new('RGB', (label_width + window * cell, top_height + 3 * cell + bottom_height), 'white')
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(25, bold=True)
    label_font = load_font(24, bold=True)
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
        draw.multiline_text((x + 60, top_height + 3 * cell + 5), text, fill='black', font=step_font, spacing=1)
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


def main():
    args = parse_args()
    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    summary = {
        task: rank_task(input_root, output_root, task, args.window, args.topk)
        for task in args.tasks
    }
    (output_root / 'ranking_summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps({task: len(items) for task, items in summary.items()}, indent=2))


if __name__ == '__main__':
    main()
