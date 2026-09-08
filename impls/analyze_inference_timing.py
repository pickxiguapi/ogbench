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
        cem_config = result['cem']
        if not result['use_subgoal']:
            if int(cem_config['iterations']) == 30:
                method = 'LeWM'
            else:
                method = (
                    f"LeWM (CEM{int(cem_config['num_samples'])}×"
                    f"{int(cem_config['iterations'])})"
                )
        elif result['cem']['cost_mode'] == 'moh':
            method = 'LeWM++'
        else:
            method = 'LeWM++ w/o MoH'
        end_to_end = timing['end_to_end']
        modules = timing['modules']
        plan_steps = end_to_end.get(
            'plan_step_samples', end_to_end.get('replan_step_samples')
        )
        first_plan = plan_steps[0]
        row = {
            'method': method,
            'generator_family': _family(result),
            'task': result['task'],
            'horizon': int(result['goal_offset_steps']),
            'eval_budget': int(result['eval_budget']),
            'num_eval': int(result['num_eval']),
            'eval_seed': int(result['seed']),
            'cem_num_samples': int(cem_config['num_samples']),
            'cem_iterations': int(cem_config['iterations']),
            'cem_horizon': int(cem_config['horizon']),
            'plan_events': int(
                timing['counts'].get(
                    'plan_events', timing['counts'].get('replan_events')
                )
            ),
            # Backward-compatible read of result JSON written before the
            # reporting unit was standardized to one complete plan.
            'steady_plan_ms': float(
                end_to_end['steady_plan_ms']
                if 'steady_plan_ms' in end_to_end
                else end_to_end['steady_replan_ms_per_environment']
            ),
            'cold_plan_ms': (
                float(first_plan['elapsed_seconds']) * 1_000.0
                / int(first_plan.get('plans', first_plan.get('replans')))
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
        row['other_plan_ms'] = row['steady_plan_ms'] - component_sum
        row['subgoal_generator_ms'] = row['subgoal_total_mean_ms']
        row['policy_ms'] = row['action_prior_mean_ms']
        row['planning_ms'] = row['steady_plan_ms'] - sum(
            value or 0.0
            for value in (row['subgoal_generator_ms'], row['policy_ms'])
        )
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
        for config_key in ('cem_num_samples', 'cem_iterations', 'cem_horizon'):
            values = {row[config_key] for row in group}
            if len(values) != 1:
                raise RuntimeError(
                    f'Mixed {config_key} in group: {method}, {family}, H{horizon}'
                )
            item[config_key] = values.pop()
        keys = (
            'steady_plan_ms',
            'cold_plan_ms',
            'subgoal_generator_ms',
            'policy_ms',
            'planning_ms',
            'subgoal_encoder_mean_ms',
            'latent_path_flow_mean_ms',
            'subgoal_total_mean_ms',
            'action_prior_mean_ms',
            'cem_mean_ms',
            'other_plan_ms',
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
        '- Exact CEM sample/iteration counts and planner horizons are read from each result file and shown below.',
        '- LeWM++ additionally uses LatentPathFlow Euler16 and a shared-all Action Prior. A matched terminal-cost run isolates the MoH timing increment.',
        '- H25 uses `goalmax25`; H50 uses `general_uniform_future`. H75/H100 use the same general-family inference graph and therefore have the same per-decision compute shape as H50.',
        '- Steady-state numbers exclude JAX compilation. `±` below is sample standard deviation across the four tasks, not uncertainty across random seeds.',
        '',
        '## Macro results',
        '',
        '| Method | Family | Goal H | CEM config | Planner H | Steady plan (ms/plan) | Cold first plan (ms/plan) |',
        '|---|---|---:|---:|---:|---:|---:|',
    ]
    for row in aggregate_rows:
        lines.append(
            f"| {row['method']} | {row['generator_family']} | {row['horizon']} | "
            f"{row['cem_num_samples']}×{row['cem_iterations']} | "
            f"{row['cem_horizon']} | "
            f"{_fmt(row['steady_plan_ms_macro_mean'])} ± {_fmt(row['steady_plan_ms_across_task_std'])} | "
            f"{_fmt(row['cold_plan_ms_macro_mean'])} |"
        )

    lines.extend(
        [
            '',
            '## LeWM++ module breakdown',
            '',
            'The three top-level modules below are mutually exclusive and sum to the complete plan time.',
            '',
            '| Variant | Family | Goal H | Subgoal Generator | Policy | Planning | Total plan |',
            '|---|---|---:|---:|---:|---:|---:|',
        ]
    )
    for row in aggregate_rows:
        if not row['method'].startswith('LeWM++'):
            continue
        lines.append(
            f"| {row['method']} | {row['generator_family']} | {row['horizon']} | "
            f"{_fmt(row['subgoal_generator_ms_macro_mean'])} | "
            f"{_fmt(row['policy_ms_macro_mean'])} | "
            f"{_fmt(row['planning_ms_macro_mean'])} | "
            f"{_fmt(row['steady_plan_ms_macro_mean'])} |"
        )

    lines.extend(
        [
            '',
            'Subgoal Generator includes history/goal encoding, LatentPathFlow, and generator-side runtime. Policy is the Action Prior actor call. Planning includes the CEM core plus planner-side key construction, warm start, array conversion, and action scaling.',
            '',
            '### Diagnostic sub-breakdown',
            '',
            '| Variant | Family | Goal H | Subgoal encoder | LatentPathFlow | CEM core | Planner runtime |',
            '|---|---|---:|---:|---:|---:|---:|',
        ]
    )
    for row in aggregate_rows:
        if not row['method'].startswith('LeWM++'):
            continue
        lines.append(
            f"| {row['method']} | {row['generator_family']} | {row['horizon']} | "
            f"{_fmt(row['subgoal_encoder_mean_ms_macro_mean'])} | "
            f"{_fmt(row['latent_path_flow_mean_ms_macro_mean'])} | "
            f"{_fmt(row['cem_mean_ms_macro_mean'])} | "
            f"{_fmt(row['other_plan_ms_macro_mean'])} |"
        )

    lines.extend(
        [
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
            '| Method | Family | H | Task | Subgoal Generator | Policy | Planning | Total plan | Plans |',
            '|---|---|---:|---|---:|---:|---:|---:|---:|',
        ]
    )
    for row in sorted(rows, key=lambda item: (item['horizon'], item['method'], item['task'])):
        lines.append(
            f"| {row['method']} | {row['generator_family']} | {row['horizon']} | "
            f"{row['task']} | {_fmt(row['subgoal_generator_ms'])} | "
            f"{_fmt(row['policy_ms'])} | {_fmt(row['planning_ms'])} | "
            f"{_fmt(row['steady_plan_ms'])} | {row['plan_events']} |"
        )

    findings = []
    for horizon, family in ((25, 'goalmax25'), (50, 'general_uniform_future')):
        lewm = lookup.get(('LeWM', 'none', horizon))
        lewmpp = lookup.get(('LeWM++', family, horizon))
        if lewm is None or lewmpp is None:
            continue
        speedup = (
            lewm['steady_plan_ms_macro_mean']
            / lewmpp['steady_plan_ms_macro_mean']
        )
        subgoal_share = (
            lewmpp['subgoal_generator_ms_macro_mean']
            / lewmpp['steady_plan_ms_macro_mean']
            * 100.0
        )
        policy_share = (
            lewmpp['policy_ms_macro_mean']
            / lewmpp['steady_plan_ms_macro_mean']
            * 100.0
        )
        planning_share = (
            lewmpp['planning_ms_macro_mean']
            / lewmpp['steady_plan_ms_macro_mean']
            * 100.0
        )
        findings.append(
            f"H{horizon}: LeWM++ is {speedup:.2f}× faster per plan than LeWM. "
            f"Within LeWM++, Subgoal Generator, Policy, and Planning are "
            f"{subgoal_share:.1f}%, {policy_share:.1f}%, and "
            f"{planning_share:.1f}% of plan wall time."
        )
        no_moh = lookup.get(('LeWM++ w/o MoH', family, horizon))
        if no_moh is not None:
            moh_delta = (
                lewmpp['cem_mean_ms_macro_mean']
                - no_moh['cem_mean_ms_macro_mean']
            )
            findings.append(
                f"H{horizon}: the matched MoH reduction changes "
                f"CEM latency by {moh_delta:+.3f} ms per plan relative to "
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
            '- The comparison unit is one complete planning call (`ms/plan`); no action-level amortization or throughput is reported. Environment rendering and stepping are excluded.',
            '- The three reported modules are exhaustive by definition: `Planning = Total plan - Subgoal Generator - Policy`. Its diagnostic sub-breakdown is CEM core plus measured planner runtime.',
            '- Cold-start latency is the first vectorized plan batch divided by its number of plan calls; it documents compilation cost but is not a single-environment startup benchmark.',
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
