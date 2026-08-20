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
    parser.add_argument('--cem-num-samples', type=int, default=300)
    parser.add_argument('--cem-steps', type=int, default=30)
    parser.add_argument('--cem-topk', type=int, default=30)
    parser.add_argument('--output', required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)
    env = ogbench.make_env_and_datasets(args.env_name, env_only=True)
    env.reset(seed=args.seed)
    scaler = NPZActionScaler(args.dataset_path)
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
        var_scale=1.0,
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
        'method': 'lewm_jax_cem',
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
            'history_len': 1,
        },
        'metrics': metrics,
        'overall_success': float(np.mean(all_successes)),
        'evaluation_time': time.time() - started,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(json_safe(result), indent=2) + '\n')
    print(json.dumps(json_safe(result), indent=2))


if __name__ == '__main__':
    main()
