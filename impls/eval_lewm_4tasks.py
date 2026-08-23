"""Evaluate policy-only, LeWM-only, and guided control on LeWM-4Tasks."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from gciql_chunk_policy import GCIQLChunkPolicy, load_agent_config, load_lance_policy
from lewm_jax.planner import JAXLeWMCEMPolicy, json_safe
from ogbench.lewm_envs.evaluation import (
    HDF5EvaluationDataset,
    StandardActionScaler,
    evaluate_dataset_goals,
    task_paths,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=('cube', 'pusht', 'reacher', 'tworoom'), required=True)
    parser.add_argument('--mode', choices=('policy', 'lewm', 'guided', 'native_q'), required=True)
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
    parser.add_argument('--cem-steps', type=int, default=5)
    parser.add_argument('--cem-topk', type=int, default=30)
    parser.add_argument('--cem-var-scale', type=float, default=1.0)
    parser.add_argument('--proposal-num-samples', type=int, default=64)
    parser.add_argument('--proposal-temperature', type=float, default=0.1)
    parser.add_argument('--video-dir')
    parser.add_argument('--output', required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    needs_lewm = args.mode != 'policy'
    needs_policy = args.mode != 'lewm'
    if needs_lewm != (args.lewm_checkpoint is not None):
        raise ValueError('This mode has an invalid --lewm-checkpoint combination.')
    if needs_policy != (args.policy_checkpoint_dir is not None):
        raise ValueError('This mode has an invalid --policy-checkpoint-dir combination.')

    hdf5_path, lance_path = task_paths(args.task, args.data_root)
    dataset = HDF5EvaluationDataset(hdf5_path)
    try:
        episodes, starts = dataset.sample_starts(
            args.num_eval, args.goal_offset_steps, args.seed
        )
        scaler = StandardActionScaler(dataset.get_column('action'))
        proposal_agent = None
        representation_mode = None
        if needs_policy:
            _, _, policy_flags = load_agent_config(args.policy_checkpoint_dir)
            representation_mode = policy_flags['representation']['mode']
            proposal_agent = load_lance_policy(
                lance_path,
                args.policy_checkpoint_dir,
                args.policy_checkpoint_step,
            )
        if args.mode == 'policy':
            policy = GCIQLChunkPolicy(proposal_agent, scaler, args.seed)
        else:
            selection = 'native_q' if args.mode == 'native_q' else 'mode'
            policy = JAXLeWMCEMPolicy(
                args.lewm_checkpoint,
                scaler,
                seed=args.seed,
                horizon=args.cem_horizon,
                receding_horizon=args.cem_receding_horizon,
                action_block=args.action_block,
                num_samples=args.cem_num_samples,
                steps=args.cem_steps,
                topk=args.cem_topk,
                var_scale=args.cem_var_scale,
                cost_mode='min_over_horizon',
                proposal_agent=proposal_agent,
                proposal_temperature=(
                    args.proposal_temperature if args.mode == 'native_q' else 0.0
                ),
                proposal_num_samples=(
                    args.proposal_num_samples if args.mode == 'native_q' else 1
                ),
                proposal_selection=selection,
                paired_plan_keys=True,
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
        'mode': args.mode,
        'representation_mode': representation_mode,
        'lewm_checkpoint': args.lewm_checkpoint,
        'policy_checkpoint_dir': args.policy_checkpoint_dir,
        'policy_checkpoint_step': args.policy_checkpoint_step if needs_policy else None,
        'seed': args.seed,
        'num_eval': args.num_eval,
        'goal_offset_steps': args.goal_offset_steps,
        'eval_budget': args.eval_budget,
        'cem': (
            None
            if args.mode == 'policy'
            else {
                'horizon': args.cem_horizon,
                'receding_horizon': args.cem_receding_horizon,
                'action_block': args.action_block,
                'num_samples': args.cem_num_samples,
                'steps': args.cem_steps,
                'topk': args.cem_topk,
                'var_scale': args.cem_var_scale,
                'cost_mode': 'min_over_horizon',
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
