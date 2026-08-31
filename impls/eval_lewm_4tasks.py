"""Evaluate policy-only, LeWM-only, and guided control on LeWM-4Tasks."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from gciql_chunk_policy import (
    GCIQLChunkPolicy,
    LatentSubgoalGCIQLChunkPolicy,
    load_agent_config,
    load_lance_policy,
)
from lewm_jax.planner import JAXLeWMCEMPolicy

from ogbench.lewm_envs.evaluation import (
    HDF5EvaluationDataset,
    StandardActionScaler,
    evaluate_dataset_goals,
    json_safe,
    task_paths,
)

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=('cube', 'pusht', 'reacher', 'tworoom'), required=True)
    parser.add_argument(
        '--controller', choices=('direct_policy', 'lewm_cem'), required=True
    )
    parser.add_argument(
        '--policy-guidance', choices=('none', 'mode'), default='none'
    )
    parser.add_argument('--use-subgoal', action='store_true')
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--lewm-checkpoint')
    parser.add_argument('--policy-checkpoint-dir')
    parser.add_argument('--policy-checkpoint-step', type=int, default=100_000)
    parser.add_argument('--num-eval', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--goal-offset-steps', type=int, default=25)
    parser.add_argument('--eval-budget', type=int, default=50)
    parser.add_argument('--cem-horizon', type=int, default=5)
    parser.add_argument('--cem-receding-horizon', type=int, default=1)
    parser.add_argument('--action-block', type=int, default=5)
    parser.add_argument('--cem-num-samples', type=int, default=300)
    parser.add_argument('--cem-iterations', type=int, default=30)
    parser.add_argument('--cem-topk', type=int, default=30)
    parser.add_argument('--cem-var-scale', type=float, default=1.0)
    parser.add_argument('--latent-subgoal-checkpoint')
    parser.add_argument('--num-samples', type=int, default=1)
    parser.add_argument(
        '--cem-cost-mode',
        choices=('last', 'moh'),
        default='moh',
    )
    parser.add_argument('--video-dir')
    parser.add_argument('--output', required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    needs_lewm = args.controller == 'lewm_cem'
    needs_policy = args.controller == 'direct_policy' or args.policy_guidance != 'none'
    needs_subgoal = args.use_subgoal
    if args.controller == 'direct_policy' and args.policy_guidance != 'none':
        raise ValueError('Policy guidance only applies to the lewm_cem controller.')
    if needs_lewm != (args.lewm_checkpoint is not None):
        raise ValueError('Invalid controller/--lewm-checkpoint combination.')
    if needs_policy != (args.policy_checkpoint_dir is not None):
        raise ValueError('Invalid controller/guidance policy-checkpoint combination.')
    if needs_subgoal != (args.latent_subgoal_checkpoint is not None):
        raise ValueError(
            'Invalid use-subgoal/--latent-subgoal-checkpoint combination.'
        )
    if args.num_samples <= 0:
        raise ValueError('--num-samples must be positive.')
    if not needs_subgoal and args.num_samples != 1:
        raise ValueError('--num-samples only applies when --use-subgoal is set.')

    hdf5_path, lance_path = task_paths(args.task, args.data_root)
    dataset = HDF5EvaluationDataset(hdf5_path)
    try:
        episodes, starts = dataset.sample_starts(
            args.num_eval, args.goal_offset_steps, args.seed
        )
        scaler = StandardActionScaler(dataset.get_column('action'))
        policy_agent = None
        representation_mode = None
        if needs_policy:
            _, _, policy_flags = load_agent_config(args.policy_checkpoint_dir)
            representation_mode = policy_flags.get('representation', {}).get(
                'mode', 'independent'
            )
            policy_agent = load_lance_policy(
                lance_path,
                args.policy_checkpoint_dir,
                args.policy_checkpoint_step,
            )
            if (
                needs_subgoal
                and args.policy_guidance != 'none'
                and representation_mode not in ('pi', 'all')
            ):
                raise ValueError(
                    'Subgoal-guided CEM requires a pi/all checkpoint whose actor '
                    'accepts frozen LeWM latent goals.'
                )
        if args.controller == 'direct_policy':
            if needs_subgoal:
                policy = LatentSubgoalGCIQLChunkPolicy(
                    policy_agent,
                    scaler,
                    args.seed,
                    args.latent_subgoal_checkpoint,
                    args.num_samples,
                    args.action_block,
                )
            else:
                policy = GCIQLChunkPolicy(policy_agent, scaler, args.seed)
        else:
            policy = JAXLeWMCEMPolicy(
                args.lewm_checkpoint,
                scaler,
                seed=args.seed,
                horizon=args.cem_horizon,
                receding_horizon=args.cem_receding_horizon,
                action_block=args.action_block,
                num_samples=args.cem_num_samples,
                iterations=args.cem_iterations,
                topk=args.cem_topk,
                var_scale=args.cem_var_scale,
                cost_mode=args.cem_cost_mode,
                guidance_policy=policy_agent,
                paired_plan_keys=True,
                latent_subgoal_checkpoint=args.latent_subgoal_checkpoint,
                latent_subgoal_num_samples=args.num_samples,
            )
        started = time.time()
        metrics = evaluate_dataset_goals(
            task=args.task,
            dataset=dataset,
            episodes=episodes,
            starts=starts,
            goal_offset=args.goal_offset_steps,
            eval_budget=args.eval_budget,
            policy=policy,
            video_dir=args.video_dir,
        )
    finally:
        dataset.close()

    result = {
        'suite': 'lewm_4tasks',
        'task': args.task,
        'controller': args.controller,
        'policy_guidance': args.policy_guidance,
        'use_subgoal': args.use_subgoal,
        'representation_mode': representation_mode,
        'lewm_checkpoint': args.lewm_checkpoint,
        'policy_checkpoint_dir': args.policy_checkpoint_dir,
        'policy_checkpoint_step': args.policy_checkpoint_step if needs_policy else None,
        'latent_subgoal': (
            None
            if not needs_subgoal
            else {
                'checkpoint': policy.latent_subgoal_checkpoint,
                'lewm_checkpoint': policy.lewm_checkpoint,
                'checkpoint_step': policy.latent_subgoal_checkpoint_step,
                'num_samples': policy.latent_subgoal_num_samples,
                'sample_selection': policy.latent_subgoal_sample_selection,
                'training_subgoal_steps': int(
                    policy.latent_subgoal_config['subgoal_steps']
                ),
                'training_action_block': int(
                    policy.latent_subgoal_config['action_block']
                ),
                'selected_waypoint_index': policy.latent_subgoal_waypoint_index,
                'selected_waypoint_step': policy.latent_subgoal_waypoint_step,
                'history_size': policy.latent_subgoal_history_size,
                'generation_counts': policy.latent_subgoal_generation_counts,
            }
        ),
        'seed': args.seed,
        'num_eval': args.num_eval,
        'goal_offset_steps': args.goal_offset_steps,
        'eval_budget': args.eval_budget,
        'cem': (
            None
            if args.controller == 'direct_policy'
            else {
                'requested_horizon': args.cem_horizon,
                'horizon': policy.horizon,
                'receding_horizon': args.cem_receding_horizon,
                'action_block': args.action_block,
                'num_samples': args.cem_num_samples,
                'iterations': args.cem_iterations,
                'topk': args.cem_topk,
                'var_scale': args.cem_var_scale,
                'cost_mode': args.cem_cost_mode,
            }
        ),
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
