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
        self.dataset_path = str(dataset_path)
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

    def sample_action_blocks(
        self,
        block_size,
        num_blocks,
        seed,
        plan_horizon=1,
        return_context_pixels=False,
    ):
        with np.load(self.dataset_path) as archive:
            actions = archive['actions'].astype(np.float32, copy=False)
            terminals = archive['terminals'].astype(bool, copy=False)
        invalid = terminals | np.isnan(actions).any(axis=1)
        prefix = np.concatenate(([0], np.cumsum(invalid, dtype=np.int64)))
        span = block_size * plan_horizon
        valid_starts = np.flatnonzero(prefix[span:] - prefix[:-span] == 0)
        if not len(valid_starts):
            raise ValueError('Dataset has no valid empirical action blocks.')
        rng = np.random.default_rng(seed)
        starts = rng.choice(
            valid_starts,
            size=num_blocks,
            replace=len(valid_starts) < num_blocks,
        )
        blocks = np.stack(
            [actions[starts + offset] for offset in range(span)], axis=1
        )
        blocks = self.transform(blocks).reshape(
            num_blocks, plan_horizon, block_size * self.action_dim
        )
        if plan_horizon == 1:
            blocks = blocks[:, 0]
        blocks = blocks.astype(np.float32)
        if not return_context_pixels:
            return blocks
        with np.load(self.dataset_path) as archive:
            context_pixels = archive['observations'][starts]
        return blocks, context_pixels


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
    parser.add_argument('--planner-history-size', type=int, default=1)
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
        '--cem-temporal-parameterization',
        choices=('independent', 'constant'),
        default='independent',
    )
    parser.add_argument(
        '--cem-empirical-action-reservoir-size', type=int, default=0
    )
    parser.add_argument('--cem-empirical-full-plans', action='store_true')
    parser.add_argument('--cem-empirical-state-conditioned', action='store_true')
    parser.add_argument('--cem-return-best-candidate', action='store_true')
    parser.add_argument('--latent-probe-qpos-indices')
    parser.add_argument('--latent-probe-samples', type=int, default=20_000)
    parser.add_argument('--latent-probe-ridge', type=float, default=1e-3)
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
    parser.add_argument('--video-dir')
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


def fit_latent_probe(policy, dataset_path, qpos_indices, num_samples, ridge, seed):
    if num_samples < 100:
        raise ValueError('Latent probe requires at least 100 samples.')
    if ridge < 0:
        raise ValueError('Latent probe ridge coefficient cannot be negative.')
    with np.load(dataset_path) as archive:
        total = len(archive['terminals'])
        rng = np.random.default_rng(seed)
        sample_indices = rng.choice(total, size=min(num_samples, total), replace=False)
        observations = archive['observations'][sample_indices]
        targets = archive['qpos'][sample_indices][:, qpos_indices].astype(np.float32)

    embeddings = []
    for offset in range(0, len(observations), 512):
        embeddings.append(policy.encode_pixels(observations[offset : offset + 512]))
    embeddings = np.concatenate(embeddings, axis=0).astype(np.float32)

    order = rng.permutation(len(embeddings))
    split = max(int(len(order) * 0.8), 1)
    train_indices, test_indices = order[:split], order[split:]
    train_x, train_y = embeddings[train_indices], targets[train_indices]
    x_mean = train_x.mean(axis=0)
    y_mean = train_y.mean(axis=0)
    y_scale = train_y.std(axis=0, ddof=1)
    y_scale = np.where(y_scale > 1e-6, y_scale, 1.0)
    centered_x = train_x - x_mean
    standardized_y = (train_y - y_mean) / y_scale
    covariance = centered_x.T @ centered_x / len(centered_x)
    cross_covariance = centered_x.T @ standardized_y / len(centered_x)
    weight = np.linalg.solve(
        covariance + ridge * np.eye(covariance.shape[0], dtype=np.float32),
        cross_covariance,
    ).astype(np.float32)
    bias = (-x_mean @ weight).astype(np.float32)

    test_predictions = embeddings[test_indices] @ weight + bias
    test_targets = (targets[test_indices] - y_mean) / y_scale
    residual = np.sum((test_predictions - test_targets) ** 2, axis=0)
    total_variance = np.sum(
        (test_targets - test_targets.mean(axis=0)) ** 2, axis=0
    )
    r2 = 1.0 - residual / np.maximum(total_variance, 1e-12)
    return weight, bias, {
        'qpos_indices': list(qpos_indices),
        'num_samples': len(embeddings),
        'ridge': ridge,
        'target_mean': y_mean,
        'target_scale': y_scale,
        'test_r2': r2,
    }


def main():
    args = parse_args()
    if args.cem_empirical_action_reservoir_size < 0:
        raise ValueError('Empirical action reservoir size cannot be negative.')
    if (args.proposal_method is None) != (args.proposal_checkpoint_dir is None):
        raise ValueError(
            '--proposal-method and --proposal-checkpoint-dir must be provided together.'
        )
    if args.cem_empirical_action_reservoir_size and args.proposal_method is not None:
        raise ValueError('Empirical action initialization cannot use a policy proposal.')
    if args.latent_probe_qpos_indices and args.proposal_method is not None:
        raise ValueError('Latent-probe cost cannot use a policy proposal.')
    np.random.seed(args.seed)
    env = ogbench.make_env_and_datasets(args.env_name, env_only=True)
    env.reset(seed=args.seed)
    scaler = NPZActionScaler(args.dataset_path)
    empirical_action_blocks = None
    empirical_context_pixels = None
    if args.cem_empirical_action_reservoir_size:
        empirical_result = scaler.sample_action_blocks(
            args.action_block,
            args.cem_empirical_action_reservoir_size,
            args.seed,
            args.cem_horizon if args.cem_empirical_full_plans else 1,
            args.cem_empirical_state_conditioned,
        )
        if args.cem_empirical_state_conditioned:
            empirical_action_blocks, empirical_context_pixels = empirical_result
        else:
            empirical_action_blocks = empirical_result
    elif args.cem_empirical_state_conditioned:
        raise ValueError(
            'State-conditioned empirical plans require a nonzero reservoir size.'
        )
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
        temporal_parameterization=args.cem_temporal_parameterization,
        empirical_action_blocks=empirical_action_blocks,
        context_history_size=args.planner_history_size,
        return_best_candidate=args.cem_return_best_candidate,
    )
    latent_probe_info = None
    if empirical_context_pixels is not None:
        context_embeddings = []
        for offset in range(0, len(empirical_context_pixels), 512):
            context_embeddings.append(
                policy.encode_pixels(empirical_context_pixels[offset : offset + 512])
            )
        policy.set_empirical_context_embeddings(
            np.concatenate(context_embeddings, axis=0)
        )
    if args.latent_probe_qpos_indices:
        qpos_indices = tuple(
            int(value) for value in args.latent_probe_qpos_indices.split(',')
        )
        weight, bias, latent_probe_info = fit_latent_probe(
            policy,
            args.dataset_path,
            qpos_indices,
            args.latent_probe_samples,
            args.latent_probe_ridge,
            args.seed,
        )
        policy.set_latent_probe(weight, bias)

    task_infos = env.unwrapped.task_infos
    metrics = {}
    all_successes = []
    all_actions = []
    episode_diagnostics = []
    video_dir = Path(args.video_dir) if args.video_dir else None
    if video_dir is not None:
        video_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    try:
        for task_id, task_info in enumerate(task_infos, start=1):
            successes = []
            for episode_index in trange(args.num_eval, desc=task_info['task_name']):
                observation, info = env.reset(options={'task_id': task_id})
                goal = np.asarray(info['goal'], dtype=np.uint8)
                initial_qpos = np.asarray(info['qpos']).copy()
                initial_privileged_positions = {
                    key: np.asarray(value).copy()
                    for key, value in info.items()
                    if key.startswith('privileged/')
                    and key.endswith('_pos')
                    and np.asarray(value).shape == (3,)
                }
                previous_effector = np.asarray(info['proprio/effector_pos']).copy()
                effector_path_length = 0.0
                episode_actions = []
                video_frames = [np.asarray(observation)] if video_dir is not None else None
                policy.reset(env.action_space, num_envs=1)
                done = False
                while not done:
                    action = policy.get_actions(
                        np.asarray(observation, dtype=np.uint8)[None, None],
                        goal[None, None],
                        np.asarray([True]),
                    )[0]
                    action = np.clip(action, env.action_space.low, env.action_space.high)
                    episode_actions.append(np.asarray(action).copy())
                    observation, _, terminated, truncated, info = env.step(action)
                    effector = np.asarray(info['proprio/effector_pos'])
                    effector_path_length += float(
                        np.linalg.norm(effector - previous_effector)
                    )
                    previous_effector = effector.copy()
                    if video_frames is not None:
                        video_frames.append(np.asarray(observation))
                    done = terminated or truncated
                success = float(info['success'])
                successes.append(success)
                episode_actions = np.asarray(episode_actions)
                all_actions.append(episode_actions)
                episode_diagnostics.append(
                    {
                        'task': task_info['task_name'],
                        'episode': episode_index,
                        'success': success,
                        'num_steps': len(episode_actions),
                        'mean_abs_action': float(np.mean(np.abs(episode_actions))),
                        'saturated_action_fraction': float(
                            np.mean(
                                np.isclose(episode_actions, env.action_space.low)
                                | np.isclose(episode_actions, env.action_space.high)
                            )
                        ),
                        'effector_path_length': effector_path_length,
                        'qpos_displacement': float(
                            np.linalg.norm(np.asarray(info['qpos']) - initial_qpos)
                        ),
                        'privileged_position_displacements': {
                            key: float(
                                np.linalg.norm(np.asarray(info[key]) - initial_value)
                            )
                            for key, initial_value in initial_privileged_positions.items()
                        },
                    }
                )
                if video_frames is not None:
                    import imageio.v2 as imageio

                    imageio.mimsave(
                        video_dir
                        / f"{task_id:02d}_{task_info['task_name']}_ep{episode_index}.mp4",
                        video_frames,
                        fps=20,
                    )
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
            'history_len': args.planner_history_size,
            'execution_steps': policy.execution_steps,
            'paired_plan_keys': args.paired_plan_keys,
            'environment_action_bounds_enforced_during_planning': True,
            'temporal_parameterization': args.cem_temporal_parameterization,
            'empirical_action_reservoir_size': (
                args.cem_empirical_action_reservoir_size
            ),
            'empirical_full_plans': args.cem_empirical_full_plans,
            'empirical_state_conditioned': args.cem_empirical_state_conditioned,
            'return_best_candidate': args.cem_return_best_candidate,
        },
        'metrics': metrics,
        'overall_success': float(np.mean(all_successes)),
        'latent_probe': latent_probe_info,
        'action_diagnostics': {
            'mean_abs_action': float(
                np.mean(np.abs(np.concatenate(all_actions, axis=0)))
            ),
            'saturated_action_fraction': float(
                np.mean(
                    np.isclose(
                        np.concatenate(all_actions, axis=0), env.action_space.low
                    )
                    | np.isclose(
                        np.concatenate(all_actions, axis=0), env.action_space.high
                    )
                )
            ),
            'episodes': episode_diagnostics,
        },
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
