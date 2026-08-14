"""Evaluate a JAX LeWM checkpoint with CEM in OGBench's built-in LeWM environments."""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import deque
from pathlib import Path

import flax
import jax
import jax.numpy as jnp
import numpy as np

from lewm_jax import ARCHITECTURE, LeWM
from ogbench.lewm_envs.evaluation import (
    HDF5EvaluationDataset,
    StandardActionScaler,
    evaluate_dataset_goals,
    task_paths,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=('cube', 'pusht', 'tworoom', 'reacher'), required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--num-eval', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--goal-offset-steps', type=int, default=25)
    parser.add_argument('--eval-budget', type=int, default=50)
    parser.add_argument('--cem-horizon', type=int, default=5)
    parser.add_argument('--cem-receding-horizon', type=int, default=5)
    parser.add_argument('--action-block', type=int, default=5)
    parser.add_argument('--cem-num-samples', type=int, default=300)
    parser.add_argument('--cem-steps', type=int, default=30)
    parser.add_argument('--cem-topk', type=int, default=30)
    parser.add_argument('--cem-var-scale', type=float, default=1.0)
    parser.add_argument('--video-dir')
    parser.add_argument('--output', required=True)
    return parser.parse_args()


def checkpoint_epoch(path):
    match = re.search(r'weights_epoch_(\d+)\.msgpack$', str(path))
    if match is None:
        raise ValueError(f'Cannot infer epoch from checkpoint path: {path}')
    return int(match.group(1))


class JAXLeWMCEMPolicy:
    """Direct JAX CEM planner with normalized action-block warm starts."""

    def __init__(
        self,
        checkpoint,
        scaler,
        *,
        seed,
        horizon,
        receding_horizon,
        action_block,
        num_samples,
        steps,
        topk,
        var_scale,
    ):
        if horizon <= 0 or receding_horizon <= 0 or action_block <= 0:
            raise ValueError('CEM horizon, receding horizon, and action block must be positive.')
        if receding_horizon > horizon:
            raise ValueError('CEM receding horizon cannot exceed the planning horizon.')
        if not 1 < topk <= num_samples:
            raise ValueError('CEM topk must be in [2, num_samples].')

        payload = flax.serialization.msgpack_restore(Path(checkpoint).read_bytes())
        config = payload['config']
        if config.get('architecture') != ARCHITECTURE:
            raise ValueError(f'Checkpoint architecture {config.get("architecture")!r} is not {ARCHITECTURE!r}.')
        precision = config.get('precision', 'bf16')
        if precision == 'bf16':
            model_dtype = jnp.bfloat16
        elif precision == 'float32':
            model_dtype = jnp.float32
        else:
            raise ValueError(f'Unsupported checkpoint precision: {precision!r}.')

        self.model = LeWM(
            image_size=int(config['image_size']),
            embed_dim=int(config['embed_dim']),
            history_size=int(config['history_size']),
            projector_hidden_dim=int(config.get('projector_hidden_dim', 2048)),
            action_smoothed_dim=int(config.get('action_smoothed_dim', 10)),
            action_mlp_scale=int(config.get('action_mlp_scale', 4)),
            predictor_depth=int(config.get('predictor_depth', 6)),
            predictor_heads=int(config.get('predictor_heads', 16)),
            predictor_dim_head=int(config.get('predictor_dim_head', 64)),
            predictor_mlp_dim=int(config.get('predictor_mlp_dim', 2048)),
            predictor_dropout=float(config.get('predictor_dropout', 0.1)),
            predictor_emb_dropout=float(config.get('predictor_emb_dropout', 0.0)),
            dtype=model_dtype,
        )
        self.variables = {'params': payload['params'], 'batch_stats': payload['batch_stats']}
        self.checkpoint_metadata = {
            'epoch': int(payload['epoch']),
            'architecture': config['architecture'],
            'encoder': config.get('encoder', 'impala_small'),
            'precision': precision,
            'image_size': int(config['image_size']),
        }
        self.scaler = scaler
        self.rng = jax.random.PRNGKey(seed)
        self.horizon = int(horizon)
        self.receding_horizon = int(receding_horizon)
        self.action_block = int(action_block)
        self.num_samples = int(num_samples)
        self.steps = int(steps)
        self.topk = int(topk)
        self.var_scale = float(var_scale)
        self._plan_one = jax.jit(self._build_plan_one())

    def _build_plan_one(self):
        model = self.model
        variables = self.variables
        num_samples = self.num_samples
        steps = self.steps
        topk = self.topk
        var_scale = self.var_scale

        def plan_one(key, pixels, goals, initial_mean):
            std = jnp.full_like(initial_mean, var_scale)

            def cem_step(_, carry):
                key, mean, std = carry
                key, sample_key = jax.random.split(key)
                candidates = (
                    jax.random.normal(
                        sample_key,
                        (num_samples, initial_mean.shape[0], initial_mean.shape[1]),
                        dtype=jnp.float32,
                    )
                    * std[None]
                    + mean[None]
                )
                candidates = candidates.at[0].set(mean)
                costs = model.apply(
                    variables,
                    pixels[None, None],
                    goals[None, None],
                    candidates[None],
                    method=model.rollout_cost,
                )[0]
                _, elite_indices = jax.lax.top_k(-costs, topk)
                elites = candidates[elite_indices]
                return key, elites.mean(axis=0), elites.std(axis=0, ddof=1)

            _, mean, std = jax.lax.fori_loop(0, steps, cem_step, (key, initial_mean, std))
            return mean, std

        return plan_one

    def reset(self, action_space, num_envs):
        action_dim = int(np.prod(action_space.shape))
        if action_dim != self.scaler.action_dim:
            raise ValueError(f'Environment action dim {action_dim} differs from dataset dim {self.scaler.action_dim}.')
        self.atomic_action_dim = action_dim
        self.block_action_dim = action_dim * self.action_block
        self.buffers = [deque() for _ in range(num_envs)]
        self.warm_starts = [None] * num_envs

    def _initial_mean(self, env_index):
        initial = np.zeros((self.horizon, self.block_action_dim), dtype=np.float32)
        warm = self.warm_starts[env_index]
        if warm is not None:
            initial[: len(warm)] = warm
        return initial

    def get_actions(self, pixels, goals, alive):
        for env_index in np.flatnonzero(alive):
            if self.buffers[env_index]:
                continue
            self.rng, plan_key = jax.random.split(self.rng)
            normalized_blocks, _ = self._plan_one(
                plan_key,
                jnp.asarray(pixels[env_index]),
                jnp.asarray(goals[env_index]),
                jnp.asarray(self._initial_mean(env_index)),
            )
            normalized_blocks = np.asarray(normalized_blocks)
            keep = normalized_blocks[: self.receding_horizon]
            self.warm_starts[env_index] = normalized_blocks[self.receding_horizon :].copy()
            normalized_atomic = keep.reshape(-1, self.atomic_action_dim)
            atomic = self.scaler.inverse_transform(normalized_atomic)
            self.buffers[env_index].extend(atomic)

        actions = np.full((len(alive), self.atomic_action_dim), np.nan, dtype=np.float32)
        for env_index in np.flatnonzero(alive):
            actions[env_index] = self.buffers[env_index].popleft()
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
    hdf5_path, _ = task_paths(args.task, args.data_root)
    dataset = HDF5EvaluationDataset(hdf5_path)
    try:
        episodes, starts = dataset.sample_starts(args.num_eval, args.goal_offset_steps, args.seed)
        scaler = StandardActionScaler(dataset.get_column('action'))
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
            image_size=policy.checkpoint_metadata['image_size'],
            video_dir=args.video_dir,
        )
        elapsed = time.time() - started
    finally:
        dataset.close()

    result = {
        'task': args.task,
        'method': 'lewm_jax_cem',
        'environment_source': 'ogbench.lewm_envs',
        'encoder': policy.checkpoint_metadata['encoder'],
        'architecture': policy.checkpoint_metadata['architecture'],
        'precision': policy.checkpoint_metadata['precision'],
        'checkpoint': args.checkpoint,
        'checkpoint_step': checkpoint_epoch(args.checkpoint),
        'seed': args.seed,
        'num_eval': args.num_eval,
        'goal_offset_steps': args.goal_offset_steps,
        'eval_budget': args.eval_budget,
        'cem': {
            'horizon': args.cem_horizon,
            'receding_horizon': args.cem_receding_horizon,
            'action_block': args.action_block,
            'num_samples': args.cem_num_samples,
            'steps': args.cem_steps,
            'topk': args.cem_topk,
            'var_scale': args.cem_var_scale,
            'batch_size': 1,
            'history_len': 1,
            'warm_start': True,
        },
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
