"""Evaluate a LeWM-JAX checkpoint in an OGBench visual environment."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import ogbench
from tqdm import trange

from eval_lewm_jax_cem import JAXLeWMCEMPolicy, json_safe


class NPZActionScaler:
    """Match the action normalization used by LeWMNPZSequenceDataset."""

    def __init__(self, dataset_path):
        with np.load(dataset_path) as archive:
            actions = archive['actions']
            terminals = archive['terminals'].astype(bool, copy=False)
        actions = actions[~terminals]
        actions = actions[~np.isnan(actions).any(axis=1)]
        self.mean = actions.mean(axis=0)
        self.scale = actions.std(axis=0, ddof=1)
        self.scale = np.where(self.scale > 0, self.scale, 1.0)
        self.action_dim = int(actions.shape[-1])

    def inverse_transform(self, value):
        return np.asarray(value) * self.scale + self.mean

    def transform(self, value):
        return (np.asarray(value) - self.mean) / self.scale


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--env-name', required=True)
    parser.add_argument('--dataset-path', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--num-eval', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cem-horizon', type=int, default=5)
    parser.add_argument('--cem-receding-horizon', type=int, default=5)
    parser.add_argument('--action-block', type=int, default=1)
    parser.add_argument(
        '--execution-steps',
        type=int,
        help='Atomic actions to execute before replanning (default: full receding horizon).',
    )
    parser.add_argument('--cem-num-samples', type=int, default=300)
    parser.add_argument('--cem-steps', type=int, default=30)
    parser.add_argument('--cem-topk', type=int, default=30)
    parser.add_argument('--cem-var-scale', type=float, default=1.0)
    parser.add_argument(
        '--cem-cost-mode',
        choices=('terminal', 'min_over_horizon'),
        default='terminal',
    )
    parser.add_argument('--proposal-method', choices=('gciql_chunk',))
    parser.add_argument('--proposal-checkpoint-dir')
    parser.add_argument('--proposal-checkpoint-step', type=int, default=500000)
    parser.add_argument('--proposal-temperature', type=float, default=0.0)
    parser.add_argument(
        '--proposal-action-space',
        choices=('planner', 'environment'),
        default='planner',
    )
    parser.add_argument('--proposal-population-size', type=int, default=0)
    parser.add_argument('--proposal-num-samples', type=int, default=1)
    parser.add_argument(
        '--proposal-selection',
        choices=('mode', 'lewm', 'lewm_cem', 'native_q'),
        default='mode',
    )
    parser.add_argument('--proposal-elite-size', type=int, default=1)
    parser.add_argument('--proposal-residual-weight', type=float, default=1.0)
    parser.add_argument('--paired-plan-keys', action='store_true')
    parser.add_argument('--native-q-keep', type=int, default=0)
    parser.add_argument('--output', required=True)
    return parser.parse_args()


def load_visual_proposal_agent(env, checkpoint_dir, checkpoint_step):
    """Restore a visual GCIQL-Chunk agent without materializing its full dataset."""
    from agents import agents
    from eval_ogbench_agent_lewm_envs import agent_config
    from utils.flax_utils import restore_agent

    config = agent_config('gciql_chunk', checkpoint_dir)
    observation = np.zeros(
        (1, *env.observation_space.shape), dtype=env.observation_space.dtype
    )
    action_width = int(np.prod(env.action_space.shape)) * int(config.chunk_size)
    actions = np.zeros((1, action_width), dtype=np.float32)
    agent = agents[config.agent_name].create(0, observation, actions, config)
    return restore_agent(agent, checkpoint_dir, checkpoint_step)


def main():
    args = parse_args()
    if (args.proposal_method is None) != (args.proposal_checkpoint_dir is None):
        raise ValueError(
            '--proposal-method and --proposal-checkpoint-dir must be provided together.'
        )
    np.random.seed(args.seed)
    env = ogbench.make_env_and_datasets(args.env_name, env_only=True)
    env.reset(seed=args.seed)
    scaler = NPZActionScaler(args.dataset_path)
    proposal_agent = None
    if args.proposal_method is not None:
        proposal_agent = load_visual_proposal_agent(
            env, args.proposal_checkpoint_dir, args.proposal_checkpoint_step
        )
    policy = JAXLeWMCEMPolicy(
        args.checkpoint,
        scaler,
        seed=args.seed,
        horizon=args.cem_horizon,
        receding_horizon=args.cem_receding_horizon,
        action_block=args.action_block,
        num_samples=args.cem_num_samples,
        steps=args.cem_steps,
        topk=args.cem_topk,
        var_scale=args.cem_var_scale,
        cost_mode=args.cem_cost_mode,
        proposal_agent=proposal_agent,
        proposal_temperature=args.proposal_temperature,
        proposal_action_space=args.proposal_action_space,
        proposal_num_samples=args.proposal_num_samples,
        proposal_population_size=args.proposal_population_size,
        proposal_selection=args.proposal_selection,
        proposal_elite_size=args.proposal_elite_size,
        proposal_residual_weight=args.proposal_residual_weight,
        native_q_keep=args.native_q_keep,
        paired_plan_keys=args.paired_plan_keys,
        execution_steps=args.execution_steps,
        action_low=env.action_space.low,
        action_high=env.action_space.high,
    )

    task_infos = env.unwrapped.task_infos
    metrics = {}
    all_successes = []
    started = time.time()
    try:
        for task_id, task_info in enumerate(task_infos, start=1):
            successes = []
            for _ in trange(args.num_eval, desc=task_info['task_name']):
                observation, info = env.reset(options={'task_id': task_id})
                goal = np.asarray(info['goal'], dtype=np.uint8)
                policy.reset(env.action_space, num_envs=1)
                done = False
                while not done:
                    action = policy.get_actions(
                        np.asarray(observation, dtype=np.uint8)[None, None],
                        goal[None, None],
                        np.asarray([True]),
                    )[0]
                    action = np.clip(action, env.action_space.low, env.action_space.high)
                    observation, _, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                successes.append(float(info['success']))
            score = float(np.mean(successes))
            metrics[task_info['task_name']] = score
            all_successes.extend(successes)
            print(json.dumps({'task': task_info['task_name'], 'success': score}), flush=True)
    finally:
        env.close()

    result = {
        'environment': args.env_name,
        'method': (
            'lewm_jax_cem'
            if args.proposal_method is None
            else f'lewm_jax_cem_{args.proposal_method}_proposal'
        ),
        'checkpoint': args.checkpoint,
        'seed': args.seed,
        'episodes_per_task': args.num_eval,
        'num_tasks': len(task_infos),
        'cem': {
            'horizon': args.cem_horizon,
            'receding_horizon': args.cem_receding_horizon,
            'action_block': args.action_block,
            'num_samples': args.cem_num_samples,
            'steps': args.cem_steps,
            'topk': args.cem_topk,
            'var_scale': args.cem_var_scale,
            'cost_mode': args.cem_cost_mode,
            'history_len': 1,
            'execution_steps': policy.execution_steps,
            'paired_plan_keys': args.paired_plan_keys,
            'environment_action_bounds_enforced_during_planning': True,
        },
        'metrics': metrics,
        'overall_success': float(np.mean(all_successes)),
        'evaluation_time': time.time() - started,
        'proposal': (
            None
            if args.proposal_method is None
            else {
                'method': args.proposal_method,
                'checkpoint_dir': args.proposal_checkpoint_dir,
                'checkpoint_step': args.proposal_checkpoint_step,
                'temperature': args.proposal_temperature,
                'injection': 'first_block_initial_mean',
                'action_space': args.proposal_action_space,
                'num_samples': args.proposal_num_samples,
                'selection': args.proposal_selection,
                'elite_size': args.proposal_elite_size,
                'residual_weight': args.proposal_residual_weight,
                'population_injection_size': args.proposal_population_size,
                'native_q_keep': args.native_q_keep,
            }
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(json_safe(result), indent=2) + '\n')
    print(json.dumps(json_safe(result), indent=2))


if __name__ == '__main__':
    main()
