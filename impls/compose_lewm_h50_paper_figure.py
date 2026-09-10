"""Compose compact side-by-side PushT and Cube H50 qualitative panels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from rank_lewm_figure_candidates import crop_cell


COLORS = {
    'real': '#222222',
    'imagined': '#0072B2',
    'subgoal': '#D55E00',
    'goal': '#009E73',
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-root', required=True)
    parser.add_argument('--ranking-root', required=True)
    parser.add_argument('--output-root', required=True)
    parser.add_argument('--pusht-rank', type=int, default=2)
    parser.add_argument('--cube-rank', type=int, default=2)
    parser.add_argument('--dpi', type=int, default=300)
    return parser.parse_args()


def load_candidate(input_root, ranking_root, task, rank):
    ranking_path = ranking_root / task / 'ranking.json'
    ranking = json.loads(ranking_path.read_text())
    matches = [item for item in ranking if int(item['rank']) == rank]
    if len(matches) != 1:
        raise ValueError(f'Expected exactly one {task} rank {rank} in {ranking_path}.')
    candidate = matches[0]
    trace_path = input_root / task / 'trace' / candidate['source_trace']
    source_path = input_root / task / 'figures' / candidate['source_figure']
    with np.load(trace_path, allow_pickle=False) as trace:
        frames = np.asarray(trace['frames'], dtype=np.uint8)
        goal = np.asarray(trace['goal'], dtype=np.uint8)
    source_rows = dict(zip(candidate['target_steps'], candidate['source_rows']))
    # crop_cell uses PIL because the source renderer has a fixed pixel layout.
    from PIL import Image

    composite_pil = Image.open(source_path).convert('RGB')
    rows = []
    for row_index in range(3):
        cells = []
        for step in candidate['display_steps']:
            if row_index == 0 or (row_index == 1 and step == 0):
                cells.append(frames[step])
            elif step == 0:
                cells.append(None)
            else:
                cells.append(np.asarray(crop_cell(composite_pil, source_rows[step], row_index)))
        rows.append(cells)
    return candidate, rows, goal


def add_cell(fig, bounds, image, border_color, blank=False):
    axis = fig.add_axes(bounds)
    axis.set_xticks([])
    axis.set_yticks([])
    if blank:
        axis.set_facecolor('#FAFAFA')
    else:
        axis.imshow(image, interpolation='nearest')
    for spine in axis.spines.values():
        spine.set_color(border_color)
        spine.set_linewidth(1.15)
    return axis


def main():
    args = parse_args()
    input_root = Path(args.input_root)
    ranking_root = Path(args.ranking_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            'font.family': 'serif',
            'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
            'pdf.fonttype': 42,
            'ps.fonttype': 42,
        }
    )
    selected = {
        'pusht': load_candidate(input_root, ranking_root, 'pusht', args.pusht_rank),
        'cube': load_candidate(input_root, ranking_root, 'cube', args.cube_rank),
    }

    figure = plt.figure(figsize=(7.35, 2.34), facecolor='white')
    left = 0.105
    right = 0.012
    panel_gap = 0.026
    panel_width = (1.0 - left - right - panel_gap) / 2.0
    bottom = 0.125
    row_height = 0.229
    cell_width = panel_width / 6.0

    row_specs = [
        ('Real', 'real'),
        ('LeWM\nImagined', 'imagined'),
        ('Predicted\nSubgoal', 'subgoal'),
    ]
    for row_index, (label, color_name) in enumerate(row_specs):
        y = bottom + (2 - row_index) * row_height
        figure.text(
            0.012,
            y + row_height / 2,
            label,
            ha='left',
            va='center',
            fontsize=8.2,
            fontweight=('bold' if row_index else 'normal'),
            color=COLORS[color_name],
            linespacing=0.9,
        )

    panel_metadata = {}
    for panel_index, (task, panel_label) in enumerate((('pusht', '(a) PushT'), ('cube', '(b) Cube'))):
        candidate, rows, goal = selected[task]
        panel_left = left + panel_index * (panel_width + panel_gap)
        figure.text(
            panel_left,
            0.955,
            panel_label,
            ha='left',
            va='top',
            fontsize=9.0,
            fontweight='bold',
        )
        goal_width = cell_width * 0.55
        goal_height = goal_width * 7.35 / 2.34
        goal_left = panel_left + panel_width - goal_width
        goal_bottom = 0.840
        goal_axis = figure.add_axes([goal_left, goal_bottom, goal_width, goal_height])
        goal_axis.imshow(goal, interpolation='nearest')
        goal_axis.set_xticks([])
        goal_axis.set_yticks([])
        for spine in goal_axis.spines.values():
            spine.set_color(COLORS['goal'])
            spine.set_linewidth(1.3)
        figure.text(
            goal_left - 0.006,
            goal_bottom + goal_height / 2,
            'Task Goal',
            ha='right',
            va='center',
            fontsize=7.2,
            color=COLORS['goal'],
            fontweight='bold',
        )

        for row_index, (_, color_name) in enumerate(row_specs):
            y = bottom + (2 - row_index) * row_height
            for column, image in enumerate(rows[row_index]):
                x = panel_left + column * cell_width
                add_cell(
                    figure,
                    [x, y, cell_width, row_height],
                    image,
                    COLORS[color_name],
                    blank=image is None,
                )
        for column, step in enumerate(candidate['display_steps']):
            x = panel_left + (column + 0.5) * cell_width
            label = f'$t={step}$'
            if step == 0:
                label += '\nContext'
            figure.text(x, 0.093, label, ha='center', va='top', fontsize=6.7, linespacing=0.9)

        panel_metadata[task] = {
            key: candidate[key]
            for key in (
                'rank',
                'episode_index',
                'episode',
                'start',
                'success',
                'mean_imagination_mse',
                'mean_subgoal_mse',
                'pusht_blue_displacement_px',
                'pusht_blue_path_length_px',
                'display_steps',
                'plan_steps',
                'target_steps',
                'source_trace',
            )
        }

    stem = 'lewm_h50_pusht_cube_six_frame_with_goals'
    png_path = output_root / f'{stem}.png'
    pdf_path = output_root / f'{stem}.pdf'
    figure.savefig(png_path, dpi=args.dpi, facecolor='white')
    figure.savefig(pdf_path, dpi=args.dpi, facecolor='white')
    plt.close(figure)
    metadata = {
        'protocol': (
            'Six frames t={0,10,20,30,40,50}. At t=0, Real and LeWM show the '
            'observed context and the subgoal cell is blank. At t>=10, both predicted '
            'rows are aligned to the real frame at the same target time and were '
            'predicted from t-10. Goal insets are the fixed H50 observations.'
        ),
        'input_root': str(input_root.resolve()),
        'ranking_root': str(ranking_root.resolve()),
        'panels': panel_metadata,
        'outputs': [png_path.name, pdf_path.name],
    }
    (output_root / f'{stem}.json').write_text(json.dumps(metadata, indent=2))
    (output_root / 'latex_include.tex').write_text(
        '\\begin{figure*}[t]\n'
        '  \\centering\n'
        f'  \\includegraphics[width=0.95\\textwidth]{{figures/{pdf_path.name}}}\n'
        '  \\caption{Qualitative H50 predictions for PushT and Cube. The top row '
        'shows real observations, the middle row LeWM imagination, and the bottom '
        'row decoded predicted subgoals. The first column is observed context; each '
        'later prediction is made ten steps earlier. Insets show the fixed goal.}\n'
        '  \\label{fig:lewm_h50_qualitative}\n'
        '\\end{figure*}\n'
    )
    print(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    main()
