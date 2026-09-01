"""Evaluate policy-only, LeWM-only, and guided control on OGBench-Env-8Tasks."""

from __future__ import annotations

import argparse
import json
import time
from collections import deque
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import ogbench
from tqdm import trange

from gciql_chunk_policy import LeWMEncodedAgent, load_agent_config
from lewm_jax import load_frozen_lewm
from lewm_jax.planner import JAXLeWMCEMPolicy
from ogbench.lewm_envs.evaluation import json_safe


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


class OGBenchChunkPolicy:
    """Execute a GCIQL-Chunk policy directly in an OGBench environment."""

    def __init__(self, agent, scaler, action_space, seed):
        self.agent = agent
        self.scaler = scaler
        self.action_space = action_space
        self.rng = jax.random.PRNGKey(seed)
        self.action_horizon = int(agent.action_horizon)

    def reset(self, action_space, num_envs):
        self.action_dim = int(np.prod(action_space.shape))
        self.buffers = [deque() for _ in range(num_envs)]

    def get_actions(self, pixels, goals, alive):
        for index in np.flatnonzero(alive):
            if self.buffers[index]:
                continue
            self.rng, key = jax.random.split(self.rng)
            chunk = np.asarray(
                self.agent.sample_actions(
                    observations=np.asarray(pixels[index, -1:]),
                    goals=np.asarray(goals[index, -1:]),
                    seed=key,
                    temperature=0.0,
                )
            )[0].reshape(self.action_horizon, self.action_dim)
            if self.action_space == 'planner':
                chunk = self.scaler.inverse_transform(chunk)
            self.buffers[index].extend(chunk)
        actions = np.full((len(alive), self.action_dim), np.nan, dtype=np.float32)
        for index in np.flatnonzero(alive):
            actions[index] = self.buffers[index].popleft()
        return actions


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--env-name', required=True)
    parser.add_argument('--dataset-path', required=True)
    parser.add_argument(
        '--controller', choices=('direct_policy', 'lewm_cem'), required=True
    )
    parser.add_argument(
        '--policy-guidance', choices=('none', 'mode'), default='none'
    )
    parser.add_argument('--lewm-checkpoint')
    parser.add_argument('--policy-checkpoint-dir')
    parser.add_argument('--policy-checkpoint-step', type=int, default=500_000)
    parser.add_argument('--policy-action-space', choices=('environment', 'planner'), default='environment')
    parser.add_argument('--num-eval', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cem-horizon', type=int, default=5)
    parser.add_argument('--cem-receding-horizon', type=int, default=1)
    parser.add_argument('--action-block', type=int, default=5)
    parser.add_argument('--cem-num-samples', type=int, default=300)
    parser.add_argument('--cem-iterations', type=int, default=30)
    parser.add_argument('--cem-topk', type=int, default=30)
    parser.add_argument('--cem-var-scale', type=float, default=1.0)
    parser.add_argument('--cem-cost-mode', choices=('last', 'moh'), default='moh')
    parser.add_argument('--video-dir')
    parser.add_argument('--output', required=True)
    return parser.parse_args()


def load_policy(env, checkpoint_dir, checkpoint_step):
    from agents import agents
    from agents.gciql_chunk_lewm import LeWMGCIQLChunkAgent
    from utils.flax_utils import restore_agent

    name, config, saved = load_agent_config(checkpoint_dir)
    observation = np.zeros(
        (1, *env.observation_space.shape), dtype=env.observation_space.dtype
    )
    action_width = int(np.prod(env.action_space.shape)) * int(config.chunk_size)
    actions = np.zeros((1, action_width), dtype=np.float32)
    if name == 'gciql_chunk':
        agent = agents[name].create(0, observation, actions, config)
        return restore_agent(agent, checkpoint_dir, checkpoint_step), saved

    lewm_checkpoint = saved.get('lewm_checkpoint')
    if lewm_checkpoint is None:
        lewm_checkpoint = saved['representation']['lewm_checkpoint']
    model, variables, metadata = load_frozen_lewm(lewm_checkpoint)
    agent = LeWMGCIQLChunkAgent.create(
        0,
        observation,
        np.zeros((1, int(metadata['config']['embed_dim'])), dtype=np.float32),
        actions,
        config,
    )
    agent = restore_agent(agent, checkpoint_dir, checkpoint_step)
    encode_pixels = jax.jit(
        lambda pixels: model.apply(
            variables, pixels, train=False, method=model.encode_pixels
        ).astype(jnp.float32)
    )
    return (
        LeWMEncodedAgent(
            agent,
            encode_pixels,
            share_pi_encoder=config.share_pi_encoder,
            lewm_checkpoint=metadata['path'],
        ),
        saved,
    )


def main():
    args = parse_args()
    needs_lewm = args.controller == 'lewm_cem'
    needs_policy = args.controller == 'direct_policy' or args.policy_guidance != 'none'
    if args.controller == 'direct_policy' and args.policy_guidance != 'none':
        raise ValueError('Policy guidance only applies to the lewm_cem controller.')
    if needs_lewm != (args.lewm_checkpoint is not None):
        raise ValueError('Invalid controller/--lewm-checkpoint combination.')
    if needs_policy != (args.policy_checkpoint_dir is not None):
        raise ValueError('Invalid controller/guidance policy-checkpoint combination.')

    np.random.seed(args.seed)
    env = ogbench.make_env_and_datasets(args.env_name, env_only=True)
    env.reset(seed=args.seed)
    scaler = NPZActionScaler(args.dataset_path)
    policy_agent = None
    representation_mode = None
    if needs_policy:
        policy_agent, policy_flags = load_policy(
            env,
            args.policy_checkpoint_dir,
            args.policy_checkpoint_step,
        )
        representation_mode = policy_flags['representation']['mode']
    if args.controller == 'direct_policy':
        policy = OGBenchChunkPolicy(
            policy_agent, scaler, args.policy_action_space, args.seed
        )
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
            guidance_mode=args.policy_guidance,
            guidance_action_space=args.policy_action_space,
            paired_plan_keys=True,
            action_low=env.action_space.low,
            action_high=env.action_space.high,
        )

    task_infos = env.unwrapped.task_infos
    metrics = {}
    all_successes = []
    video_dir = Path(args.video_dir) if args.video_dir else None
    if video_dir is not None:
        video_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    try:
        for task_id, task_info in enumerate(task_infos, start=1):
            successes = []
            for episode in trange(args.num_eval, desc=task_info['task_name']):
                observation, info = env.reset(options={'task_id': task_id})
                goal = np.asarray(info['goal'], dtype=np.uint8)
                frames = [np.asarray(observation)] if video_dir is not None else None
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
                    if frames is not None:
                        frames.append(np.asarray(observation))
                    done = terminated or truncated
                successes.append(float(info['success']))
                if frames is not None:
                    import imageio.v2 as imageio

                    imageio.mimsave(
                        video_dir / f'{task_id:02d}_{task_info["task_name"]}_ep{episode}.mp4',
                        frames,
                        fps=20,
                    )
            metrics[task_info['task_name']] = float(np.mean(successes))
            all_successes.extend(successes)
    finally:
        env.close()

    result = {
        'suite': 'ogbench_env_8tasks',
        'environment': args.env_name,
        'controller': args.controller,
        'policy_guidance': args.policy_guidance,
        'representation_mode': representation_mode,
        'lewm_checkpoint': args.lewm_checkpoint,
        'policy_checkpoint_dir': args.policy_checkpoint_dir,
        'policy_checkpoint_step': args.policy_checkpoint_step if needs_policy else None,
        'policy_action_space': args.policy_action_space if needs_policy else None,
        'cem': (
            None
            if args.controller == 'direct_policy'
            else {
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
        'seed': args.seed,
        'episodes_per_task': args.num_eval,
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
