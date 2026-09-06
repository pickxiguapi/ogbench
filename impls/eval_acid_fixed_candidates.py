"""Evaluate ACID consistency on fixed CEM candidates with simulator outcomes."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np

from acid_idm import load_acid_idm_checkpoint, sample_inverse_actions
from acid_metrics import (
    binary_auc,
    correlation,
    grouped_risk_at_coverages,
    grouped_upper_tail_auc,
    safe_mean,
    safe_std,
)
from gciql_chunk_policy import load_agent_config, load_lance_policy
from lewm_jax.planner import JAXLeWMCEMPolicy
from ogbench.lewm_envs.evaluation import (
    HDF5EvaluationDataset,
    StandardActionScaler,
    TASK_SPECS,
    _dataset_pixels,
    _resize_frame,
    _set_dataset_state,
    task_paths,
)


COVERAGES = (0.25, 0.50, 0.75, 1.0)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=('cube', 'pusht', 'reacher', 'tworoom'), required=True)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--lewm-checkpoint', required=True)
    parser.add_argument('--idm-checkpoint', required=True)
    parser.add_argument('--policy-checkpoint-dir', required=True)
    parser.add_argument('--policy-checkpoint-step', type=int, default=100_000)
    parser.add_argument('--latent-subgoal-checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--num-states', type=int, default=200)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--goal-offset-steps', type=int, default=50)
    parser.add_argument('--action-block', type=int, default=5)
    parser.add_argument('--cem-horizon', type=int, default=2)
    parser.add_argument('--cem-receding-horizon', type=int, default=1)
    parser.add_argument('--cem-num-samples', type=int, default=300)
    parser.add_argument('--cem-iterations', type=int, default=5)
    parser.add_argument('--cem-topk', type=int, default=30)
    parser.add_argument('--cem-var-scale', type=float, default=1.0)
    parser.add_argument('--cem-cost-mode', choices=('last', 'moh', 'path_mean'), default='moh')
    parser.add_argument('--num-subgoal-samples', type=int, default=1)
    parser.add_argument('--high-forward-error-quantile', type=float, default=0.80)
    parser.add_argument('--encode-batch-size', type=int, default=256)
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def strict_json(value):
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): strict_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [strict_json(item) for item in value]
    return value


def low_score_mask(scores, groups, coverage):
    scores = np.asarray(scores, dtype=np.float64)
    groups = np.asarray(groups)
    selected = np.zeros(len(scores), dtype=bool)
    for group in np.unique(groups):
        indices = np.flatnonzero(groups == group)
        indices = indices[np.argsort(scores[indices], kind='mergesort')]
        retained = max(1, int(np.ceil(float(coverage) * len(indices))))
        selected[indices[:retained]] = True
    return selected


def execute_candidate(env, task, init_row, goal_row, actions, *, image_size):
    seed = (
        int(np.asarray(init_row['seed']).reshape(-1)[0])
        if 'seed' in init_row
        else None
    )
    env.reset(seed=seed)
    _set_dataset_state(task, env, init_row, goal_row)
    block_frames = []
    success = False
    finished = False
    for block in actions:
        for action in block:
            if not finished:
                _, _, terminated, truncated, _ = env.step(action)
                success |= bool(terminated)
                finished = bool(terminated or truncated)
        block_frames.append(_resize_frame(env.render(), image_size))
    return np.stack(block_frames), success


def main():
    args = parse_args()
    if args.num_states <= 0 or args.encode_batch_size <= 0:
        raise ValueError('num-states and encode-batch-size must be positive.')
    if args.cem_horizon <= 0 or args.action_block <= 0:
        raise ValueError('CEM horizon and action block must be positive.')
    if not 0.0 < args.high_forward_error_quantile < 1.0:
        raise ValueError('high-forward-error-quantile must be in (0, 1).')

    hdf5_path, lance_path = task_paths(args.task, args.data_root)
    dataset = HDF5EvaluationDataset(hdf5_path)
    spec = TASK_SPECS[args.task]
    env = gym.make(
        spec.env_id,
        max_episode_steps=args.action_block * args.cem_horizon,
        render_mode='rgb_array',
        **spec.env_kwargs,
    )
    try:
        episodes, starts = dataset.sample_starts(
            args.num_states, args.goal_offset_steps, args.seed
        )
        scaler = StandardActionScaler(dataset.get_column('action'))
        _, _, policy_flags = load_agent_config(args.policy_checkpoint_dir)
        representation_mode = policy_flags.get('representation', {}).get(
            'mode', 'independent'
        )
        if representation_mode not in ('pi', 'all'):
            raise ValueError(
                'Fixed-candidate LeWM++ evaluation requires a pi/all policy.'
            )
        policy_agent = load_lance_policy(
            lance_path,
            args.policy_checkpoint_dir,
            args.policy_checkpoint_step,
        )
        planner = JAXLeWMCEMPolicy(
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
            cost_mode=args.cem_cost_mode,
            guidance_policy=policy_agent,
            guidance_mode='mode',
            guidance_population_size=0,
            guidance_temperature=1.0,
            guidance_elite_size=8,
            guidance_goal_mode='final',
            paired_plan_keys=True,
            trace_candidates=True,
            latent_subgoal_checkpoint=args.latent_subgoal_checkpoint,
            latent_subgoal_num_samples=args.num_subgoal_samples,
        )
        predictor_config = planner.latent_subgoal_config
        if predictor_config.get('goal_sampling') != 'hiql_uniform_future_same_trajectory':
            raise ValueError('H50 requires the general uniform-future generator.')
        if predictor_config.get('max_goal_steps') is not None:
            raise ValueError('H50 general generator must not cap max_goal_steps.')

        idm, idm_params, action_mean, action_std, idm_config, idm_step = (
            load_acid_idm_checkpoint(args.idm_checkpoint)
        )
        if idm_config['lewm_checkpoint_sha256'] != sha256_file(
            args.lewm_checkpoint
        ):
            raise ValueError('IDM and planner use different frozen LeWM checkpoints.')
        if int(idm_config['transition_steps']) != args.action_block:
            raise ValueError('IDM transition length must equal one LeWM action block.')
        if int(idm_config['embed_dim']) != int(planner.lewm_config['embed_dim']):
            raise ValueError('IDM and LeWM embedding dimensions differ.')
        action_mean = np.asarray(action_mean, dtype=np.float32)
        action_std = np.asarray(action_std, dtype=np.float32)

        initial_frames = []
        goal_frames = []
        init_rows = []
        goal_rows = []
        seeds = []
        for episode, start in zip(episodes, starts):
            init_row = dataset.row(episode, start)
            goal_row = dataset.row(episode, start + args.goal_offset_steps)
            seed = (
                int(np.asarray(init_row['seed']).reshape(-1)[0])
                if 'seed' in init_row
                else None
            )
            env.reset(seed=seed)
            _set_dataset_state(args.task, env, init_row, goal_row)
            initial_frames.append(_resize_frame(env.render(), 224))
            goal_frames.append(_resize_frame(_dataset_pixels(goal_row['pixels']), 224))
            init_rows.append(init_row)
            goal_rows.append(goal_row)
            seeds.append(-1 if seed is None else seed)
        initial_frames = np.stack(initial_frames)
        goal_frames = np.stack(goal_frames)

        planner.reset(env.action_space, args.num_states)
        alive = np.ones(args.num_states, dtype=bool)
        started = time.time()
        planner.get_actions(initial_frames[:, None], goal_frames[:, None], alive)
        traces = [events[0] for events in planner.latent_subgoal_trace]
        candidate_actions = np.stack(
            [event['candidate_environment_action_blocks'] for event in traces]
        ).astype(np.float32)
        predicted_paths = np.stack(
            [event['candidate_imagined_paths'] for event in traces]
        ).astype(np.float32)
        candidate_goal_costs = np.stack(
            [event['candidate_goal_costs'] for event in traces]
        ).astype(np.float32)
        current_embeddings = np.stack(
            [event['current_embedding'] for event in traces]
        ).astype(np.float32)
        expected_action_shape = (
            args.num_states,
            args.cem_num_samples,
            planner.horizon,
            args.action_block,
            scaler.action_dim,
        )
        if candidate_actions.shape != expected_action_shape:
            raise ValueError(
                f'Candidate action shape {candidate_actions.shape} != '
                f'{expected_action_shape}.'
            )
        if not np.isfinite(candidate_actions).all():
            raise FloatingPointError('Candidate actions contain non-finite values.')
        if not np.isfinite(predicted_paths).all():
            raise FloatingPointError('Candidate imagined paths contain non-finite values.')

        real_paths = np.empty_like(predicted_paths)
        real_goal_success = np.zeros(
            (args.num_states, args.cem_num_samples), dtype=bool
        )
        for state_index in range(args.num_states):
            endpoint_frames = []
            for candidate_index in range(args.cem_num_samples):
                frames, success = execute_candidate(
                    env,
                    args.task,
                    init_rows[state_index],
                    goal_rows[state_index],
                    candidate_actions[state_index, candidate_index],
                    image_size=224,
                )
                endpoint_frames.append(frames)
                real_goal_success[state_index, candidate_index] = success
            endpoint_frames = np.stack(endpoint_frames)
            flat_frames = endpoint_frames.reshape(-1, *endpoint_frames.shape[2:])
            encoded = []
            for batch_start in range(0, len(flat_frames), args.encode_batch_size):
                encoded.append(
                    planner.encode_pixels(
                        flat_frames[batch_start : batch_start + args.encode_batch_size]
                    )
                )
            real_paths[state_index] = np.concatenate(encoded).reshape(
                args.cem_num_samples, planner.horizon, -1
            )
            print(
                f'replayed state {state_index + 1}/{args.num_states} '
                f'elapsed={time.time() - started:.1f}s',
                flush=True,
            )

        normalized_actions = (
            (candidate_actions - action_mean[None, None, None, None])
            / action_std[None, None, None, None]
        ).reshape(
            args.num_states,
            args.cem_num_samples,
            planner.horizon,
            -1,
        )
        acid_block_errors = np.empty(
            (args.num_states, args.cem_num_samples, planner.horizon),
            dtype=np.float32,
        )
        for state_index in range(args.num_states):
            previous = np.concatenate(
                (
                    np.repeat(
                        current_embeddings[state_index][None, None],
                        args.cem_num_samples,
                        axis=0,
                    ),
                    predicted_paths[state_index, :, :-1],
                ),
                axis=1,
            )
            current = previous.reshape(-1, previous.shape[-1])
            next_z = predicted_paths[state_index].reshape(
                -1, predicted_paths.shape[-1]
            )
            actions = normalized_actions[state_index].reshape(
                -1, normalized_actions.shape[-1]
            )
            key = jax.random.fold_in(jax.random.PRNGKey(args.seed), state_index)
            inverse_actions = np.asarray(
                jax.device_get(
                    sample_inverse_actions(
                        idm,
                        idm_params,
                        jnp.asarray(current),
                        jnp.asarray(next_z),
                        key,
                        num_steps=1,
                    )
                )
            )
            acid_block_errors[state_index] = np.mean(
                np.square(inverse_actions - actions), axis=-1
            ).reshape(args.cem_num_samples, planner.horizon)

        forward_block_mse = np.mean(
            np.square(predicted_paths - real_paths), axis=-1
        )
        acid_mean = acid_block_errors.mean(axis=-1)
        acid_max = acid_block_errors.max(axis=-1)
        forward_mean = forward_block_mse.mean(axis=-1)
        forward_max = forward_block_mse.max(axis=-1)
        forward_endpoint = forward_block_mse[..., -1]
        state_ids = np.repeat(
            np.arange(args.num_states, dtype=np.int32), args.cem_num_samples
        )
        flat_acid_mean = acid_mean.reshape(-1)
        flat_acid_max = acid_max.reshape(-1)
        flat_forward_mean = forward_mean.reshape(-1)
        flat_forward_max = forward_max.reshape(-1)
        flat_forward_endpoint = forward_endpoint.reshape(-1)
        flat_success = real_goal_success.reshape(-1)
        high_error_auc, high_error_labels, thresholds = grouped_upper_tail_auc(
            flat_acid_max,
            flat_forward_max,
            state_ids,
            quantile=args.high_forward_error_quantile,
        )
        risk_coverage = grouped_risk_at_coverages(
            flat_acid_max, flat_forward_max, state_ids, COVERAGES
        )
        per_state_spearman = [
            correlation(acid_max[index], forward_max[index], rank=True)
            for index in range(args.num_states)
        ]
        metrics = {
            'candidate_count': len(flat_acid_max),
            'acid_mean_cost_mean': safe_mean(flat_acid_mean),
            'acid_max_cost_mean': safe_mean(flat_acid_max),
            'forward_mean_block_mse': safe_mean(flat_forward_mean),
            'forward_max_block_mse': safe_mean(flat_forward_max),
            'forward_endpoint_mse': safe_mean(flat_forward_endpoint),
            'real_goal_success_rate': safe_mean(flat_success),
            'acid_mean_vs_forward_mean_pearson': correlation(
                flat_acid_mean, flat_forward_mean
            ),
            'acid_mean_vs_forward_mean_spearman': correlation(
                flat_acid_mean, flat_forward_mean, rank=True
            ),
            'acid_max_vs_forward_max_pearson': correlation(
                flat_acid_max, flat_forward_max
            ),
            'acid_max_vs_forward_max_spearman': correlation(
                flat_acid_max, flat_forward_max, rank=True
            ),
            'acid_max_vs_forward_max_within_state_spearman_mean': safe_mean(
                per_state_spearman
            ),
            'acid_max_vs_forward_max_within_state_spearman_std': safe_std(
                per_state_spearman
            ),
            'acid_max_predicts_within_state_high_forward_error_auc': high_error_auc,
            'high_forward_error_quantile': args.high_forward_error_quantile,
        }
        for coverage in COVERAGES:
            percent = int(round(100 * coverage))
            selected = low_score_mask(flat_acid_max, state_ids, coverage)
            metrics[f'forward_max_block_mse_at_{percent}pct_acid_coverage'] = (
                risk_coverage[coverage]
            )
            metrics[f'high_forward_error_rate_at_{percent}pct_acid_coverage'] = (
                safe_mean(high_error_labels[selected])
            )
            metrics[f'real_goal_success_rate_at_{percent}pct_acid_coverage'] = (
                safe_mean(flat_success[selected])
            )

        summary = {
            'protocol': 'fixed_final_cem_candidate_pool_simulator_replay',
            'task': args.task,
            'num_states': args.num_states,
            'candidates_per_state': args.cem_num_samples,
            'candidate_pool': 'last CEM iteration before final elite refit',
            'high_forward_error_definition': (
                'within-state upper tail of max per-block latent MSE'
            ),
            'within_state_high_error_thresholds': thresholds,
            'lewm_checkpoint': str(Path(args.lewm_checkpoint).resolve()),
            'idm_checkpoint': str(Path(args.idm_checkpoint).resolve()),
            'idm_step': idm_step,
            'policy_checkpoint_dir': str(
                Path(args.policy_checkpoint_dir).resolve()
            ),
            'policy_checkpoint_step': args.policy_checkpoint_step,
            'latent_subgoal_checkpoint': str(
                Path(args.latent_subgoal_checkpoint).resolve()
            ),
            'seed': args.seed,
            'goal_offset_steps': args.goal_offset_steps,
            'cem': {
                'horizon': planner.horizon,
                'action_block': args.action_block,
                'num_samples': args.cem_num_samples,
                'iterations': args.cem_iterations,
                'topk': args.cem_topk,
                'var_scale': args.cem_var_scale,
                'cost_mode': args.cem_cost_mode,
                'policy_guidance': 'mode/final_goal',
                'subgoal_num_samples': args.num_subgoal_samples,
            },
            'elapsed_seconds': time.time() - started,
            'metrics': metrics,
        }
        output = Path(args.output).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(strict_json(summary), indent=2, sort_keys=True) + '\n'
        )
        np.savez_compressed(
            output.with_suffix('.candidates.npz'),
            episodes=np.asarray(episodes),
            starts=np.asarray(starts),
            environment_seeds=np.asarray(seeds),
            candidate_environment_action_blocks=candidate_actions,
            candidate_goal_costs=candidate_goal_costs,
            predicted_paths=predicted_paths,
            real_paths=real_paths,
            real_goal_success=real_goal_success,
            acid_block_errors=acid_block_errors,
            forward_block_mse=forward_block_mse,
        )
        print(json.dumps(strict_json(summary), indent=2, sort_keys=True))
    finally:
        env.close()
        dataset.close()


if __name__ == '__main__':
    main()
