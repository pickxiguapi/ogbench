"""Evaluate an OGBench agent checkpoint in the built-in LeWM environment suite."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax
import numpy as np

from ogbench.lewm_envs.evaluation import (
    HDF5EvaluationDataset,
    StandardActionScaler,
    evaluate_dataset_goals,
    task_paths,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=('cube', 'pusht', 'tworoom', 'reacher'), required=True)
    parser.add_argument('--method', choices=('gciql', 'gciql_chunk', 'hiql'), required=True)
    parser.add_argument('--checkpoint-dir', required=True)
    parser.add_argument('--checkpoint-step', type=int, default=100000)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--num-eval', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--goal-offset-steps', type=int, default=25)
    parser.add_argument('--eval-budget', type=int, default=50)
    parser.add_argument('--video-dir')
    parser.add_argument('--output', required=True)
    return parser.parse_args()


def agent_config(method, checkpoint_dir=None):
    if method == 'gciql':
        from agents.gciql import get_config

        config = get_config()
        config.alpha = 1.0
    elif method == 'gciql_chunk':
        from agents.gciql_chunk import get_config

        config = get_config()
    else:
        from agents.hiql import get_config

        config = get_config()
        config.high_alpha = 3.0
        config.low_actor_rep_grad = True
        config.low_alpha = 3.0
        config.subgoal_steps = 10
    config.batch_size = 256
    config.encoder = 'impala_small'
    config.p_aug = 0.5

    if checkpoint_dir is not None:
        flags_path = Path(checkpoint_dir) / 'flags.json'
        if flags_path.is_file():
            saved_agent = json.loads(flags_path.read_text()).get('agent', {})
            if saved_agent.get('agent_name', config.agent_name) != config.agent_name:
                raise ValueError(
                    f'Checkpoint agent {saved_agent.get("agent_name")} does not match requested method {method}.'
                )
            for key, value in saved_agent.items():
                if key in config:
                    config[key] = value
    return config


def load_agent(method, lance_path, checkpoint_dir, checkpoint_step):
    from agents import agents
    from utils.datasets import GCChunkDataset, GCDataset, HGCDataset
    from utils.flax_utils import restore_agent
    from utils.lewm_dataset import LeWMLanceDataset

    config = agent_config(method, checkpoint_dir)
    base = LeWMLanceDataset(lance_path, split='train', validation_fraction=0.05)
    wrapper = {'gciql': GCDataset, 'gciql_chunk': GCChunkDataset, 'hiql': HGCDataset}[method]
    dataset = wrapper(base, config, preprocess_frame_stack=False)
    example = dataset.sample(1, evaluation=True)
    agent = agents[config.agent_name].create(0, example['observations'], example['actions'], config)
    return restore_agent(agent, checkpoint_dir, checkpoint_step)


class OGBenchAgentPolicy:
    def __init__(self, agent, scaler, seed):
        self.agent = agent
        self.scaler = scaler
        self.rng = jax.random.PRNGKey(seed)
        self.action_horizon = int(getattr(agent, 'action_horizon', 1))
        if self.action_horizon < 1:
            raise ValueError(f'Policy action_horizon must be positive, got {self.action_horizon}.')
        self._chunks = None
        self._chunk_index = 0

    def reset(self, action_space, num_envs):
        self._action_dim = int(np.prod(action_space.shape))
        self._num_envs = num_envs
        self._chunks = None
        self._chunk_index = 0

    def get_actions(self, pixels, goals, alive):
        if self._chunks is None or self._chunk_index >= self.action_horizon:
            self.rng, action_rng = jax.random.split(self.rng)
            normalized = np.asarray(
                self.agent.sample_actions(
                    observations=np.asarray(pixels[:, -1]),
                    goals=np.asarray(goals[:, -1]),
                    seed=action_rng,
                    temperature=0.0,
                )
            )
            if self.action_horizon == 1:
                self._chunks = self.scaler.inverse_transform(normalized)[:, None]
            else:
                expected = self.action_horizon * self._action_dim
                if normalized.shape[-1] != expected:
                    raise ValueError(
                        f'Agent returned width {normalized.shape[-1]}, expected '
                        f'action_horizon({self.action_horizon}) * action_dim({self._action_dim}) = {expected}.'
                    )
                atomic = self.scaler.inverse_transform(normalized.reshape(-1, self._action_dim))
                self._chunks = atomic.reshape(self._num_envs, self.action_horizon, self._action_dim)
            self._chunk_index = 0
        actions = self._chunks[:, self._chunk_index].copy()
        self._chunk_index += 1
        actions[~alive] = np.nan
        return actions


def json_safe(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    return value


def main():
    args = parse_args()
    hdf5_path, lance_path = task_paths(args.task, args.data_root)
    dataset = HDF5EvaluationDataset(hdf5_path)
    try:
        episodes, starts = dataset.sample_starts(args.num_eval, args.goal_offset_steps, args.seed)
        scaler = StandardActionScaler(dataset.get_column('action'))
        agent = load_agent(args.method, lance_path, args.checkpoint_dir, args.checkpoint_step)
        policy = OGBenchAgentPolicy(agent, scaler, args.seed)
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
        elapsed = time.time() - started
    finally:
        dataset.close()

    result = {
        'task': args.task,
        'method': args.method,
        'environment_source': 'ogbench.lewm_envs',
        'checkpoint_dir': args.checkpoint_dir,
        'checkpoint_step': args.checkpoint_step,
        'seed': args.seed,
        'num_eval': args.num_eval,
        'goal_offset_steps': args.goal_offset_steps,
        'eval_budget': args.eval_budget,
        'eval_episodes': episodes,
        'eval_start_steps': starts,
        'evaluation_time': elapsed,
        'metrics': metrics,
        'success_rate': metrics['success_rate'],
        'episodes': args.num_eval,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(json_safe(result), indent=2) + '\n')
    print(json.dumps(json_safe(result), indent=2))


if __name__ == '__main__':
    main()
