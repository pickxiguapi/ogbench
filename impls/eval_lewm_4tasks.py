"""Evaluate LeWM++, its ablations, LeWM, and GCIQL-Chunk on LeWM-4Tasks."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from action_prior import FinalGoalPolicy, load_action_prior
from lewm_jax.planner import LeWMPPController
from subgoal_generators import GENERATOR_ARCHITECTURES

from ogbench.lewm_envs.evaluation import (
    HDF5EvaluationDataset,
    StandardActionScaler,
    evaluate_dataset_goals,
    json_safe,
    task_paths,
)

VARIANTS = ('full', 'no_subgoal', 'no_action_prior', 'no_moh', 'lewm', 'gciql_chunk')
DEFAULT_CEM_SAMPLES = 300
DEFAULT_CEM_ITERATIONS = 5
DEFAULT_FLOW_STEPS = 16


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=('cube', 'pusht', 'reacher', 'tworoom'), required=True)
    parser.add_argument('--variant', choices=VARIANTS, required=True)
    parser.add_argument('--experiment-group', required=True)
    parser.add_argument(
        '--generator-family',
        choices=('goalmax25', 'general_uniform_future', 'no_generator'),
        required=True,
    )
    parser.add_argument('--generator-type', choices=tuple(GENERATOR_ARCHITECTURES), default='latent_path_flow')
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--lewm-checkpoint', required=True)
    parser.add_argument('--action-prior-checkpoint-dir')
    parser.add_argument('--action-prior-checkpoint-step', type=int, default=100_000)
    parser.add_argument(
        '--action-prior-mode',
        choices=('zero', 'policy_mode', 'policy_mode_anchor'),
        default='policy_mode',
    )
    parser.add_argument('--subgoal-generator-checkpoint')
    parser.add_argument('--flow-sampling-steps', type=int, default=DEFAULT_FLOW_STEPS)
    parser.add_argument('--generator-num-samples', type=int, default=1)
    parser.add_argument('--num-eval', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--goal-offset-steps', type=int, default=25)
    parser.add_argument('--eval-budget', type=int, default=50)
    parser.add_argument('--cem-horizon', type=int, default=2)
    parser.add_argument('--cem-receding-horizon', type=int, default=1)
    parser.add_argument('--action-block', type=int, default=5)
    parser.add_argument('--cem-num-samples', type=int, default=DEFAULT_CEM_SAMPLES)
    parser.add_argument('--cem-iterations', type=int, default=DEFAULT_CEM_ITERATIONS)
    parser.add_argument('--cem-topk', type=int, default=30)
    parser.add_argument('--cem-var-scale', type=float, default=1.0)
    parser.add_argument('--cem-min-std', type=float, default=1e-3)
    parser.add_argument('--cem-cost-mode', choices=('last', 'moh'), default='moh')
    parser.add_argument('--video-dir')
    parser.add_argument('--output', required=True)
    return parser.parse_args()


def expected_components(variant):
    """Return (subgoal, action prior, CEM cost, direct policy)."""
    return {
        'full': (True, True, 'moh', False),
        'no_subgoal': (False, True, 'moh', False),
        'no_action_prior': (True, False, 'moh', False),
        'no_moh': (True, True, 'last', False),
        'lewm': (False, False, 'last', False),
        'gciql_chunk': (False, True, None, True),
    }[variant]


def validate_args(args):
    for name in (
        'action_prior_checkpoint_step',
        'flow_sampling_steps',
        'generator_num_samples',
        'num_eval',
        'goal_offset_steps',
        'eval_budget',
        'cem_horizon',
        'cem_receding_horizon',
        'action_block',
        'cem_num_samples',
        'cem_iterations',
        'cem_topk',
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f'--{name.replace("_", "-")} must be positive.')
    if args.cem_var_scale <= 0 or args.cem_min_std <= 0:
        raise ValueError('CEM variance scale and minimum std must be positive.')

    use_subgoal, use_prior, cost_mode, direct_policy = expected_components(args.variant)
    if use_subgoal != (args.subgoal_generator_checkpoint is not None):
        raise ValueError(f'Variant {args.variant} has an invalid subgoal-generator setting.')
    if use_prior != (args.action_prior_checkpoint_dir is not None):
        raise ValueError(f'Variant {args.variant} has an invalid action-prior checkpoint setting.')
    if use_prior != (args.action_prior_mode != 'zero'):
        raise ValueError(f'Variant {args.variant} has an invalid action-prior mode.')
    if direct_policy and args.action_prior_mode != 'policy_mode':
        raise ValueError('Direct GCIQL-Chunk evaluation requires action_prior_mode=policy_mode.')
    if not direct_policy and args.cem_cost_mode != cost_mode:
        raise ValueError(f'Variant {args.variant} requires CEM cost {cost_mode}.')
    if use_subgoal == (args.generator_family == 'no_generator'):
        raise ValueError('Generator-family label does not match the selected variant.')
    if not use_subgoal and args.generator_num_samples != 1:
        raise ValueError('--generator-num-samples only applies to a subgoal generator.')


def main():
    args = parse_args()
    validate_args(args)
    use_subgoal, use_prior, _, direct_policy = expected_components(args.variant)
    hdf5_path, lance_path = task_paths(args.task, args.data_root)
    dataset = HDF5EvaluationDataset(hdf5_path)
    try:
        episodes, starts = dataset.sample_starts(args.num_eval, args.goal_offset_steps, args.seed)
        scaler = StandardActionScaler(dataset.get_column('action'))
        action_prior = (
            load_action_prior(
                lance_path,
                args.action_prior_checkpoint_dir,
                args.action_prior_checkpoint_step,
                args.lewm_checkpoint,
            )
            if use_prior
            else None
        )
        if direct_policy:
            controller = FinalGoalPolicy(action_prior, scaler, args.seed)
        else:
            controller = LeWMPPController(
                checkpoint=args.lewm_checkpoint,
                scaler=scaler,
                seed=args.seed,
                horizon=args.cem_horizon,
                receding_horizon=args.cem_receding_horizon,
                action_block=args.action_block,
                num_samples=args.cem_num_samples,
                iterations=args.cem_iterations,
                topk=args.cem_topk,
                var_scale=args.cem_var_scale,
                min_std=args.cem_min_std,
                cost_mode=args.cem_cost_mode,
                action_prior=action_prior,
                action_prior_mode=args.action_prior_mode,
                paired_plan_keys=True,
                subgoal_generator_checkpoint=args.subgoal_generator_checkpoint,
                subgoal_generator_num_samples=args.generator_num_samples,
                flow_sampling_steps=args.flow_sampling_steps,
            )
            if use_subgoal:
                actual = controller.subgoal_generator.config['architecture']
                expected = GENERATOR_ARCHITECTURES[args.generator_type]
                if actual not in expected:
                    raise ValueError(
                        f'Generator type mismatch: checkpoint has {actual!r}, expected one of {expected!r}.'
                    )

        started = time.time()
        metrics = evaluate_dataset_goals(
            task=args.task,
            dataset=dataset,
            episodes=episodes,
            starts=starts,
            goal_offset=args.goal_offset_steps,
            eval_budget=args.eval_budget,
            policy=controller,
            video_dir=args.video_dir,
        )
    finally:
        dataset.close()

    generator = None if direct_policy else controller.subgoal_generator
    result = {
        'suite': 'lewm_4tasks',
        'task': args.task,
        'variant': args.variant,
        'experiment_group': args.experiment_group,
        'generator_family': args.generator_family,
        'generator_type': args.generator_type if use_subgoal else None,
        'components': {
            'subgoal_generator': use_subgoal,
            'action_prior_mode': args.action_prior_mode,
            'min_over_horizon': None if direct_policy else args.cem_cost_mode == 'moh',
            'direct_policy': direct_policy,
        },
        'lewm_checkpoint': str(Path(args.lewm_checkpoint).expanduser().resolve()),
        'action_prior_checkpoint_dir': args.action_prior_checkpoint_dir,
        'action_prior_checkpoint_step': args.action_prior_checkpoint_step if use_prior else None,
        'subgoal_generator': (
            None
            if generator is None
            else {
                'checkpoint': generator.checkpoint,
                'checkpoint_step': generator.checkpoint_step,
                'num_samples': generator.num_samples,
                'flow_sampling_steps': generator.flow_sampling_steps,
                'config': generator.config,
                'generation_counts': generator.generation_counts,
            }
        ),
        'protocol': {
            'num_eval': args.num_eval,
            'seed': args.seed,
            'goal_offset_steps': args.goal_offset_steps,
            'eval_budget': args.eval_budget,
            'cem_horizon': None if direct_policy else controller.horizon,
            'cem_receding_horizon': None if direct_policy else args.cem_receding_horizon,
            'action_block': args.action_block,
            'cem_num_samples': None if direct_policy else args.cem_num_samples,
            'cem_iterations': None if direct_policy else args.cem_iterations,
            'cem_topk': None if direct_policy else args.cem_topk,
            'cem_var_scale': None if direct_policy else args.cem_var_scale,
            'cem_min_std': None if direct_policy else args.cem_min_std,
            'cem_cost_mode': None if direct_policy else args.cem_cost_mode,
            'flow_sampling_steps': args.flow_sampling_steps if use_subgoal else None,
            'policy_goal': 'final_goal',
            'paired_plan_keys': not direct_policy,
        },
        'metrics': metrics,
        'success_rate': metrics['success_rate'],
        'evaluation_time': time.time() - started,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(json_safe(result), indent=2) + '\n')
    print(json.dumps(json_safe(result), indent=2))


if __name__ == '__main__':
    main()
