"""Aggregate synchronized LeWM/LeWM++ inference timing result JSON files."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


TASKS = ('cube', 'pusht', 'reacher', 'tworoom')
MODULES = ('subgoal_encoder', 'latent_path_flow', 'action_prior', 'cem')


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-root', required=True)
    parser.add_argument('--output-dir')
    return parser.parse_args()


def _mean(values):
    return statistics.fmean(values) if values else None


def _std(values):
    return statistics.stdev(values) if len(values) > 1 else 0.0


def _fmt(value, digits=2):
    return '—' if value is None else f'{value:.{digits}f}'


def _family(result):
    if not result['use_subgoal']:
        return 'none'
    sampling = result['latent_subgoal']['checkpoint']
    return 'goalmax25' if 'goalmax25' in sampling else 'general_uniform_future'


def load_rows(root):
    rows = []
    for path in sorted(Path(root).glob('**/result.json')):
        result = json.loads(path.read_text())
        timing = result.get('inference_timing')
        if not timing:
            continue
        if not result['use_subgoal']:
            method = 'LeWM'
        elif result['cem']['cost_mode'] == 'moh':
            method = 'LeWM++'
        else:
            method = 'LeWM++ w/o MoH'
        end_to_end = timing['end_to_end']
        modules = timing['modules']
        first_replan = end_to_end['replan_step_samples'][0]
        row = {
            'method': method,
            'generator_family': _family(result),
            'task': result['task'],
            'horizon': int(result['goal_offset_steps']),
            'eval_budget': int(result['eval_budget']),
            'num_eval': int(result['num_eval']),
            'eval_seed': int(result['seed']),
            'replan_events': int(timing['counts']['replan_events']),
            'steady_replan_ms': float(
                end_to_end['steady_replan_ms_per_environment']
            ),
            'buffer_action_ms': float(
                end_to_end['buffer_action_ms_per_environment']
            ),
            'amortized_action_ms': float(
                end_to_end['steady_amortized_ms_per_environment_action']
            ),
            'actions_per_second': float(end_to_end['steady_actions_per_second']),
            'cold_replan_ms': (
                float(first_replan['elapsed_seconds']) * 1_000.0
                / int(first_replan['replans'])
            ),
            'evaluation_time_s': float(result['evaluation_time']),
            'success_rate': float(result['success_rate']),
            'result_path': str(path.resolve()),
        }
        for module in MODULES:
            distribution = modules.get(module)
            row[f'{module}_mean_ms'] = (
                None if distribution is None else distribution['steady_mean_ms']
            )
            row[f'{module}_median_ms'] = (
                None if distribution is None else distribution['steady_median_ms']
            )
            row[f'{module}_p95_ms'] = (
                None if distribution is None else distribution['steady_p95_ms']
            )
        subgoal = modules.get('subgoal_total')
        row['subgoal_total_mean_ms'] = (
            None if subgoal is None else subgoal['steady_mean_ms']
        )
        row['subgoal_total_median_ms'] = (
            None if subgoal is None else subgoal['steady_median_ms']
        )
        row['subgoal_total_p95_ms'] = (
            None if subgoal is None else subgoal['steady_p95_ms']
        )
        component_sum = sum(
            row[f'{module}_mean_ms'] or 0.0
            for module in ('subgoal_total', 'action_prior', 'cem')
        )
        row['other_replan_ms'] = row['steady_replan_ms'] - component_sum
        rows.append(row)
    if not rows:
        raise SystemExit(f'No profiled result.json files under {root}.')
    return rows


def aggregate(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row['method'], row['generator_family'], row['horizon'])].append(row)
    aggregate_rows = []
    for (method, family, horizon), group in sorted(grouped.items()):
        if {row['task'] for row in group} != set(TASKS):
            raise RuntimeError(
                f'Incomplete four-task group: {method}, {family}, H{horizon}'
            )
        item = {
            'method': method,
            'generator_family': family,
            'horizon': horizon,
            'num_tasks': len(group),
        }
        keys = (
            'steady_replan_ms',
            'amortized_action_ms',
            'actions_per_second',
            'cold_replan_ms',
            'subgoal_encoder_mean_ms',
            'latent_path_flow_mean_ms',
            'subgoal_total_mean_ms',
            'action_prior_mean_ms',
            'cem_mean_ms',
            'other_replan_ms',
        )
        keys += tuple(
            f'{module}_{stat}_ms'
            for module in MODULES
            for stat in ('median', 'p95')
        )
        keys += ('subgoal_total_median_ms', 'subgoal_total_p95_ms')
        for key in keys:
            values = [row[key] for row in group if row[key] is not None]
            item[f'{key}_macro_mean'] = _mean(values)
            item[f'{key}_across_task_std'] = _std(values)
        aggregate_rows.append(item)
    return aggregate_rows


def write_tsv(path, rows):
    with Path(path).open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]), delimiter='\t')
        writer.writeheader()
        writer.writerows(rows)


def build_report(rows, aggregate_rows):
    lookup = {
        (row['method'], row['generator_family'], row['horizon']): row
        for row in aggregate_rows
    }
    lines = [
        '# LeWM vs. LeWM++ Inference-Time Analysis',
        '',
        '## Protocol',
        '',
        '- Hardware: one exclusive NVIDIA A800-SXM4-80GB per task process; GPU work is explicitly synchronized before each timer stops.',
        '- LeWM: canonical CEM300x30, planner H5/RH1, action block 5, MoH.',
        '- LeWM++: LatentPathFlow Euler16, shared-all Action Prior, CEM300x5, planner H2/RH1, action block 5, MoH. A matched terminal-cost run isolates the MoH timing increment.',
        '- H25 uses `goalmax25`; H50 uses `general_uniform_future`. H75/H100 use the same general-family inference graph and therefore have the same per-decision compute shape as H50.',
        '- Steady-state numbers exclude JAX compilation. `±` below is sample standard deviation across the four tasks, not uncertainty across random seeds.',
        '',
        '## Macro results',
        '',
        '| Method | Family | Goal H | Replan (ms) | Amortized/action (ms) | Actions/s | Cold first replan (ms) |',
        '|---|---|---:|---:|---:|---:|---:|',
    ]
    for row in aggregate_rows:
        lines.append(
            f"| {row['method']} | {row['generator_family']} | {row['horizon']} | "
            f"{_fmt(row['steady_replan_ms_macro_mean'])} ± {_fmt(row['steady_replan_ms_across_task_std'])} | "
            f"{_fmt(row['amortized_action_ms_macro_mean'])} ± {_fmt(row['amortized_action_ms_across_task_std'])} | "
            f"{_fmt(row['actions_per_second_macro_mean'], 1)} | "
            f"{_fmt(row['cold_replan_ms_macro_mean'])} |"
        )

    lines.extend(
        [
            '',
            '## LeWM++ module breakdown',
            '',
        '| Variant | Family | Goal H | Subgoal encoder | LatentPathFlow | Subgoal total | Action Prior | CEM | Other | Total replan |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|',
        ]
    )
    for row in aggregate_rows:
        if not row['method'].startswith('LeWM++'):
            continue
        lines.append(
            f"| {row['method']} | {row['generator_family']} | {row['horizon']} | "
            f"{_fmt(row['subgoal_encoder_mean_ms_macro_mean'])} | "
            f"{_fmt(row['latent_path_flow_mean_ms_macro_mean'])} | "
            f"{_fmt(row['subgoal_total_mean_ms_macro_mean'])} | "
            f"{_fmt(row['action_prior_mean_ms_macro_mean'])} | "
            f"{_fmt(row['cem_mean_ms_macro_mean'])} | "
            f"{_fmt(row['other_replan_ms_macro_mean'])} | "
            f"{_fmt(row['steady_replan_ms_macro_mean'])} |"
        )

    lines.extend(
        [
            '',
            'The encoder and LatentPathFlow rows below are nested inside `Subgoal total`; they must not be added to it again.',
            '',
            '## LeWM++ steady-state distributions',
            '',
            '| Family | Module | Mean (ms) | Median (ms) | P95 (ms) |',
            '|---|---|---:|---:|---:|',
        ]
    )
    module_labels = (
        ('subgoal_encoder', 'Subgoal LeWM encoder'),
        ('latent_path_flow', 'LatentPathFlow'),
        ('subgoal_total', 'Subgoal total'),
        ('action_prior', 'Action Prior'),
        ('cem', 'CEM incl. MoH'),
    )
    for row in aggregate_rows:
        if row['method'] != 'LeWM++':
            continue
        for key, label in module_labels:
            lines.append(
                f"| {row['generator_family']} | {label} | "
                f"{_fmt(row[f'{key}_mean_ms_macro_mean'])} | "
                f"{_fmt(row[f'{key}_median_ms_macro_mean'])} | "
                f"{_fmt(row[f'{key}_p95_ms_macro_mean'])} |"
            )

    lines.extend(
        [
            '',
            '## Raw task-level data',
            '',
            '| Method | Family | H | Task | Replan (ms) | Amortized/action (ms) | CEM (ms) | Subgoal total (ms) | Action Prior (ms) | Events |',
            '|---|---|---:|---|---:|---:|---:|---:|---:|---:|',
        ]
    )
    for row in sorted(rows, key=lambda item: (item['horizon'], item['method'], item['task'])):
        lines.append(
            f"| {row['method']} | {row['generator_family']} | {row['horizon']} | "
            f"{row['task']} | {_fmt(row['steady_replan_ms'])} | "
            f"{_fmt(row['amortized_action_ms'])} | {_fmt(row['cem_mean_ms'])} | "
            f"{_fmt(row['subgoal_total_mean_ms'])} | "
            f"{_fmt(row['action_prior_mean_ms'])} | {row['replan_events']} |"
        )

    findings = []
    for horizon, family in ((25, 'goalmax25'), (50, 'general_uniform_future')):
        lewm = lookup.get(('LeWM', 'none', horizon))
        lewmpp = lookup.get(('LeWM++', family, horizon))
        if lewm is None or lewmpp is None:
            continue
        speedup = (
            lewm['amortized_action_ms_macro_mean']
            / lewmpp['amortized_action_ms_macro_mean']
        )
        flow_share = (
            lewmpp['latent_path_flow_mean_ms_macro_mean']
            / lewmpp['steady_replan_ms_macro_mean']
            * 100.0
        )
        cem_share = (
            lewmpp['cem_mean_ms_macro_mean']
            / lewmpp['steady_replan_ms_macro_mean']
            * 100.0
        )
        findings.append(
            f"H{horizon}: LeWM++ is {speedup:.2f}× faster per "
            f"amortized action than LeWM. Within LeWM++, LatentPathFlow is "
            f"{flow_share:.1f}% and CEM is {cem_share:.1f}% of replan wall time."
        )
        no_moh = lookup.get(('LeWM++ w/o MoH', family, horizon))
        if no_moh is not None:
            moh_delta = (
                lewmpp['cem_mean_ms_macro_mean']
                - no_moh['cem_mean_ms_macro_mean']
            )
            findings.append(
                f"H{horizon}: the matched MoH reduction changes "
                f"CEM latency by {moh_delta:+.3f} ms per replan relative to "
                'terminal cost; values at this scale should be interpreted as '
                'fused-kernel timing, not an additive standalone module.'
            )
    lines.extend(['', '## Key findings', ''])
    lines.extend(
        [f'{index}. {finding}' for index, finding in enumerate(findings, 1)]
        or ['Timing groups are incomplete; no paired finding computed.']
    )
    lines.extend(
        [
            '',
            '## Interpretation boundaries',
            '',
            '- This compares each method in its canonical paper configuration; it is not an equal-FLOP comparison because LeWM uses 30 CEM iterations/H5 while LeWM++ uses 5 iterations/H2 plus learned modules.',
            '- Per-action latency is the decision cost amortized over the five executed actions in each chunk; environment rendering and stepping are excluded.',
            '- `Other` is measured steady-state Python/JAX glue, key construction, array conversion, warm-start update, and action inverse-scaling overhead.',
            '- Cold-start latency is the first vectorized replan batch divided by its number of active environments; it documents compilation cost but is not a single-environment startup benchmark.',
            '- The four task processes use identical GPU models and isolated devices. Across-task variation includes task action dimensionality and environment-specific policy/model execution differences.',
            '',
        ]
    )
    return '\n'.join(lines)


def main():
    args = parse_args()
    input_root = Path(args.input_root).expanduser().resolve()
    output_dir = (
        input_root if args.output_dir is None else Path(args.output_dir).expanduser().resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows(input_root)
    aggregate_rows = aggregate(rows)
    write_tsv(output_dir / 'raw_timing.tsv', rows)
    write_tsv(output_dir / 'aggregate_timing.tsv', aggregate_rows)
    (output_dir / 'report.md').write_text(build_report(rows, aggregate_rows) + '\n')
    print(f'wrote {output_dir / "raw_timing.tsv"}')
    print(f'wrote {output_dir / "aggregate_timing.tsv"}')
    print(f'wrote {output_dir / "report.md"}')


if __name__ == '__main__':
    main()
