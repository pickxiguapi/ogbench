"""Evaluate a JAX LeWM checkpoint with CEM in LeWM environments."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from collections import deque
from pathlib import Path

import flax
import jax
import jax.numpy as jnp
import numpy as np
from latent_subgoal import (
    FLOW_TRANSFORMER_ARCHITECTURE,
    LATENT_PATH_FLOW_ARCHITECTURE,
    latent_path_waypoint_steps,
    load_latent_subgoal_checkpoint,
    sample_conditional_flow_candidates,
    sample_conditional_path_flow_candidates,
    select_latent_medoid,
    select_latent_path_medoid,
)

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
    parser.add_argument('--cem-iterations', type=int, default=30)
    parser.add_argument('--cem-topk', type=int, default=30)
    parser.add_argument('--cem-var-scale', type=float, default=1.0)
    parser.add_argument(
        '--cem-cost-mode',
        choices=('last', 'moh'),
        default='last',
    )
    parser.add_argument(
        '--proposal-method',
        choices=('gciql_chunk', 'gciql_chunk_lewm'),
    )
    parser.add_argument('--proposal-checkpoint-dir')
    parser.add_argument('--proposal-checkpoint-step', type=int, default=100000)
    parser.add_argument('--proposal-temperature', type=float, default=0.0)
    parser.add_argument('--proposal-num-samples', type=int, default=1)
    parser.add_argument(
        '--proposal-population-size',
        type=int,
        default=0,
        help=(
            'If positive, inject this many actor-sampled first blocks into '
            'iteration zero before LeWM-only CEM ranking; sample zero is mode.'
        ),
    )
    parser.add_argument(
        '--proposal-selection',
        choices=('mode', 'lewm', 'lewm_cem'),
        default='mode',
    )
    parser.add_argument('--proposal-elite-size', type=int, default=1)
    parser.add_argument('--proposal-residual-weight', type=float, default=1.0)
    parser.add_argument(
        '--paired-plan-keys',
        action='store_true',
        help=(
            'Derive CEM sampling keys from (seed, environment index, replan index) '
            'so vanilla and proposal-guided runs use matched planner randomness.'
        ),
    )
    parser.add_argument(
        '--diagnose-min-horizon',
        action='store_true',
        help=(
            'Record which predicted checkpoint minimizes latent goal distance '
            'for every executed CEM plan, including per-episode statistics.'
        ),
    )
    parser.add_argument('--video-dir')
    parser.add_argument('--output', required=True)
    return parser.parse_args()


def checkpoint_epoch(path):
    match = re.search(r'weights_epoch_(\d+)\.msgpack$', str(path))
    if match is None:
        raise ValueError(f'Cannot infer epoch from checkpoint path: {path}')
    return int(match.group(1))


def sha256_file(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open('rb') as file:
        for chunk in iter(lambda: file.read(chunk_size), b''):
            digest.update(chunk)
    return digest.hexdigest()


def latent_path_waypoint_index(waypoint_steps, target_step):
    """Return the unique path-token index corresponding to a requested step."""
    waypoint_steps = tuple(int(step) for step in waypoint_steps)
    matches = [index for index, step in enumerate(waypoint_steps) if step == target_step]
    if len(matches) != 1:
        raise ValueError(
            f'Latent path must contain target step {target_step} exactly once; '
            f'got {waypoint_steps}.'
        )
    return matches[0]


def subgoal_planning_horizon(subgoal_steps, action_block):
    """Derive the number of LeWM action blocks needed to reach a subgoal."""
    if subgoal_steps <= 0 or action_block <= 0:
        raise ValueError('Subgoal steps and action block must be positive.')
    if subgoal_steps % action_block:
        raise ValueError('Subgoal steps must be divisible by the action block.')
    return subgoal_steps // action_block


def select_latent_subgoal_costs(latent_distances, cost_mode):
    """Select one cost per candidate from rollout-to-subgoal distances."""
    if cost_mode == 'last':
        return latent_distances[:, :, -1][0]
    if cost_mode == 'moh':
        return jnp.min(latent_distances, axis=-1)[0]
    raise ValueError(f'Unsupported latent subgoal cost mode: {cost_mode!r}.')


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
        iterations,
        topk,
        var_scale,
        cost_mode='last',
        proposal_agent=None,
        proposal_temperature=0.0,
        proposal_action_space='planner',
        proposal_num_samples=1,
        proposal_population_size=0,
        proposal_selection='mode',
        proposal_elite_size=1,
        proposal_residual_weight=1.0,
        paired_plan_keys=False,
        diagnose_min_horizon=False,
        action_low=None,
        action_high=None,
        temporal_parameterization='independent',
        empirical_action_blocks=None,
        context_history_size=1,
        return_best_candidate=False,
        empirical_context_rank_penalty=0.0,
        latent_subgoal_checkpoint=None,
        latent_subgoal_num_samples=1,
    ):
        if horizon <= 0 or receding_horizon <= 0 or action_block <= 0:
            raise ValueError('CEM horizon, receding horizon, and action block must be positive.')
        if cost_mode not in ('last', 'moh'):
            raise ValueError(f'Unsupported CEM cost mode: {cost_mode!r}.')
        if (action_low is None) != (action_high is None):
            raise ValueError('Action low and high bounds must be provided together.')
        if temporal_parameterization not in ('independent', 'constant'):
            raise ValueError(
                'Temporal action parameterization must be independent or constant.'
            )
        if context_history_size <= 0:
            raise ValueError('Planner context history size must be positive.')
        if empirical_context_rank_penalty < 0:
            raise ValueError('Empirical context rank penalty cannot be negative.')
        if latent_subgoal_checkpoint is not None:
            if proposal_agent is not None:
                raise ValueError(
                    'Latent subgoal planning is a pure-CEM evaluation and cannot '
                    'use policy guidance.'
                )
            if empirical_action_blocks is not None:
                raise ValueError(
                    'Latent subgoal planning cannot use empirical action initialization.'
                )
            if diagnose_min_horizon:
                raise ValueError(
                    'Latent subgoal planning does not yet expose goal-image diagnostics.'
                )
            if latent_subgoal_num_samples != 1:
                raise ValueError(
                    'Latent subgoal inference is temporarily fixed to one sample.'
                )
        if not 1 < topk <= num_samples:
            raise ValueError('CEM topk must be in [2, num_samples].')
        if proposal_num_samples < 1:
            raise ValueError('Proposal sample count must be positive.')
        if proposal_action_space not in ('planner', 'environment'):
            raise ValueError(
                'Proposal action space must be either planner or environment.'
            )
        if proposal_action_space == 'environment' and not hasattr(
            scaler, 'transform'
        ):
            raise ValueError(
                'Environment-space proposals require a scaler with transform().'
            )
        if proposal_population_size < 0 or proposal_population_size > num_samples:
            raise ValueError('Proposal population size must be in [0, num_samples].')
        if proposal_population_size and proposal_agent is None:
            raise ValueError('Proposal population injection requires a proposal agent.')
        if proposal_selection != 'mode' and proposal_agent is None:
            raise ValueError('Proposal selection requires a proposal agent.')
        if proposal_selection != 'mode' and proposal_num_samples < 2:
            raise ValueError('Proposal selection requires at least two policy samples.')
        if not 1 <= proposal_elite_size <= proposal_num_samples:
            raise ValueError('Proposal elite size must be in [1, proposal_num_samples].')
        if proposal_selection == 'lewm_cem' and proposal_elite_size < 2:
            raise ValueError('Policy-population CEM requires at least two elites.')
        if not 0.0 <= proposal_residual_weight <= 1.0:
            raise ValueError('Proposal residual weight must be in [0, 1].')
        payload = flax.serialization.msgpack_restore(Path(checkpoint).read_bytes())
        config = payload['config']
        if config.get('architecture') != ARCHITECTURE:
            raise ValueError(f'Checkpoint architecture {config.get("architecture")!r} is not {ARCHITECTURE!r}.')
        if context_history_size > int(config['history_size']):
            raise ValueError(
                'Planner context history cannot exceed the checkpoint history size.'
            )
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
            'embed_dim': int(config['embed_dim']),
        }
        self.lewm_checkpoint = str(Path(checkpoint).resolve())
        self.latent_subgoal_checkpoint = None
        self.latent_subgoal_config = None
        self.latent_subgoal_checkpoint_step = None
        self.latent_subgoal_num_samples = int(latent_subgoal_num_samples)
        self.latent_subgoal_sample_selection = None
        self._predict_latent_subgoal = None
        self._latent_subgoal_requires_rng = False
        self.latent_subgoal_waypoint_index = None
        self.latent_subgoal_waypoint_step = None
        self.latent_subgoal_history_size = 1
        if latent_subgoal_checkpoint is not None:
            (
                subgoal_model,
                subgoal_params,
                subgoal_config,
                subgoal_checkpoint_step,
            ) = load_latent_subgoal_checkpoint(latent_subgoal_checkpoint)
            if int(subgoal_config['embed_dim']) != int(config['embed_dim']):
                raise ValueError(
                    'Latent subgoal and LeWM embedding dimensions do not match.'
                )
            expected_sha = subgoal_config.get('lewm_checkpoint_sha256')
            actual_sha = sha256_file(checkpoint)
            if expected_sha != actual_sha:
                raise ValueError(
                    'Latent subgoal generator was not trained with this LeWM checkpoint: '
                    f'expected SHA-256 {expected_sha}, got {actual_sha}.'
                )
            self.latent_subgoal_checkpoint = str(
                Path(latent_subgoal_checkpoint).expanduser().resolve()
            )
            self.latent_subgoal_config = subgoal_config
            self.latent_subgoal_checkpoint_step = subgoal_checkpoint_step
            if subgoal_config['architecture'] == FLOW_TRANSFORMER_ARCHITECTURE:
                sampling_steps = int(subgoal_config['flow_sampling_steps'])
                flow_solver = str(subgoal_config['flow_solver'])
                self.latent_subgoal_sample_selection = 'single_sample'
                self._latent_subgoal_requires_rng = True
                self._predict_latent_subgoal = jax.jit(
                    lambda current, goal, rng: select_latent_medoid(
                        sample_conditional_flow_candidates(
                            subgoal_model,
                            subgoal_params,
                            current,
                            goal,
                            rng,
                            num_samples=self.latent_subgoal_num_samples,
                            num_steps=sampling_steps,
                            solver=flow_solver,
                        )
                    )
                )
            elif subgoal_config['architecture'] == LATENT_PATH_FLOW_ARCHITECTURE:
                sampling_steps = int(subgoal_config['flow_sampling_steps'])
                flow_solver = str(subgoal_config['flow_solver'])
                history_size = int(subgoal_config.get('history_size', 1))
                if history_size <= 0:
                    raise ValueError('Latent subgoal history size must be positive.')
                waypoint_step = int(subgoal_config['subgoal_steps'])
                trained_action_block = int(subgoal_config['action_block'])
                if trained_action_block != int(action_block):
                    raise ValueError(
                        'Latent subgoal and planner action blocks must match: '
                        f'{trained_action_block} != {action_block}.'
                    )
                waypoint_steps = latent_path_waypoint_steps(
                    waypoint_step, trained_action_block
                )
                waypoint_index = latent_path_waypoint_index(
                    waypoint_steps, waypoint_step
                )
                self.latent_subgoal_waypoint_index = waypoint_index
                self.latent_subgoal_waypoint_step = waypoint_step
                self.latent_subgoal_history_size = history_size
                self.latent_subgoal_sample_selection = 'single_sample'
                self._latent_subgoal_requires_rng = True
                self._predict_latent_subgoal = jax.jit(
                    lambda current, goal, rng: select_latent_path_medoid(
                        sample_conditional_path_flow_candidates(
                            subgoal_model,
                            subgoal_params,
                            current,
                            goal,
                            rng,
                            num_samples=self.latent_subgoal_num_samples,
                            num_steps=sampling_steps,
                            solver=flow_solver,
                        )
                    )[:, waypoint_index]
                )
            else:
                self._predict_latent_subgoal = jax.jit(
                    lambda current, goal: subgoal_model.apply(
                        {'params': subgoal_params}, current, goal
                    )
                )
        self.scaler = scaler
        self.seed = int(seed)
        self.rng = jax.random.PRNGKey(seed)
        requested_horizon = int(horizon)
        if self.latent_subgoal_config is None:
            effective_horizon = requested_horizon
        else:
            effective_horizon = subgoal_planning_horizon(
                int(self.latent_subgoal_config['subgoal_steps']), int(action_block)
            )
        if int(receding_horizon) > effective_horizon:
            raise ValueError('CEM receding horizon cannot exceed the planning horizon.')
        self.requested_horizon = requested_horizon
        self.horizon = effective_horizon
        self.receding_horizon = int(receding_horizon)
        self.action_block = int(action_block)
        self.context_history_size = int(context_history_size)
        self.temporal_parameterization = str(temporal_parameterization)
        self.empirical_action_blocks = None
        self.empirical_context_embeddings = None
        self.empirical_goal_embeddings = None
        self.empirical_goal_distance_weight = 1.0
        if empirical_action_blocks is not None:
            empirical_action_blocks = np.asarray(
                empirical_action_blocks, dtype=np.float32
            )
            expected_width = int(self.scaler.action_dim) * self.action_block
            valid_shape = (
                empirical_action_blocks.ndim == 2
                and empirical_action_blocks.shape[1] == expected_width
            ) or (
                empirical_action_blocks.ndim == 3
                and empirical_action_blocks.shape[1:] == (
                    self.horizon,
                    expected_width,
                )
            )
            if not valid_shape or empirical_action_blocks.shape[0] < 2:
                raise ValueError(
                    'Empirical actions must have shape '
                    f'(N >= 2, {expected_width}) or '
                    f'(N >= 2, {self.horizon}, {expected_width}); got '
                    f'{empirical_action_blocks.shape}.'
                )
            self.empirical_action_blocks = empirical_action_blocks
        self.planner_action_low = None
        self.planner_action_high = None
        if action_low is not None:
            action_low = np.asarray(action_low, dtype=np.float32)
            action_high = np.asarray(action_high, dtype=np.float32)
            expected_shape = (int(self.scaler.action_dim),)
            if action_low.shape != expected_shape or action_high.shape != expected_shape:
                raise ValueError(
                    f'Action bounds must have shape {expected_shape}; got '
                    f'{action_low.shape} and {action_high.shape}.'
                )
            if np.any(action_low >= action_high):
                raise ValueError('Each action low bound must be smaller than its high bound.')
            self.planner_action_low = np.tile(
                self.scaler.transform(action_low), self.action_block
            ).astype(np.float32)
            self.planner_action_high = np.tile(
                self.scaler.transform(action_high), self.action_block
            ).astype(np.float32)
        self.num_samples = int(num_samples)
        self.iterations = int(iterations)
        self.topk = int(topk)
        self.var_scale = float(var_scale)
        self.cost_mode = str(cost_mode)
        self.proposal_agent = proposal_agent
        self.proposal_temperature = float(proposal_temperature)
        self.proposal_action_space = str(proposal_action_space)
        self.proposal_num_samples = int(proposal_num_samples)
        self.proposal_population_size = int(proposal_population_size)
        self.proposal_selection = str(proposal_selection)
        self.proposal_elite_size = int(proposal_elite_size)
        self.proposal_residual_weight = float(proposal_residual_weight)
        self.paired_plan_keys = bool(paired_plan_keys)
        self.diagnose_min_horizon = bool(diagnose_min_horizon)
        self.return_best_candidate = bool(return_best_candidate)
        self.empirical_context_rank_penalty = float(
            empirical_context_rank_penalty
        )
        self.latent_probe_weight = None
        self.latent_probe_bias = None
        if self.context_history_size > 1 and self.receding_horizon != 1:
            raise ValueError(
                'Multi-frame planner context requires replanning after exactly '
                'one action block.'
            )
        self._plan_one = jax.jit(self._build_plan_one())
        self._score_plans = jax.jit(self._build_score_plans())
        self._plan_distances = jax.jit(self._build_plan_distances())
        self._encode_pixels_batch = jax.jit(
            lambda value: self.model.apply(
                self.variables,
                value,
                train=False,
                method=self.model.encode_pixels,
            )
        )

    def encode_pixels(self, pixels):
        return np.asarray(self._encode_pixels_batch(jnp.asarray(pixels)))

    def set_latent_probe(self, weight, bias):
        weight = np.asarray(weight, dtype=np.float32)
        bias = np.asarray(bias, dtype=np.float32)
        embed_dim = int(self.checkpoint_metadata['embed_dim'])
        if weight.ndim != 2 or weight.shape[0] != embed_dim:
            raise ValueError(
                f'Latent probe weight must have shape ({embed_dim}, K); got '
                f'{weight.shape}.'
            )
        if bias.shape != (weight.shape[1],):
            raise ValueError(
                f'Latent probe bias must have shape ({weight.shape[1]},); got '
                f'{bias.shape}.'
            )
        self.latent_probe_weight = weight
        self.latent_probe_bias = bias
        self._plan_one = jax.jit(self._build_plan_one())
        self._score_plans = jax.jit(self._build_score_plans())
        self._plan_distances = jax.jit(self._build_plan_distances())

    def set_empirical_context_embeddings(
        self, embeddings, goal_embeddings=None, goal_distance_weight=1.0
    ):
        if self.empirical_action_blocks is None:
            raise ValueError('Empirical context requires empirical action plans.')
        embeddings = np.asarray(embeddings, dtype=np.float32)
        expected_shape = (
            self.empirical_action_blocks.shape[0],
            int(self.checkpoint_metadata['embed_dim']),
        )
        if embeddings.shape != expected_shape:
            raise ValueError(
                f'Empirical context embeddings must have shape {expected_shape}; '
                f'got {embeddings.shape}.'
            )
        if len(embeddings) < self.num_samples:
            raise ValueError(
                'Empirical context reservoir cannot be smaller than the CEM population.'
            )
        self.empirical_context_embeddings = embeddings
        if goal_embeddings is not None:
            goal_embeddings = np.asarray(goal_embeddings, dtype=np.float32)
            if goal_embeddings.shape != expected_shape:
                raise ValueError(
                    f'Empirical goal embeddings must have shape {expected_shape}; '
                    f'got {goal_embeddings.shape}.'
                )
            if goal_distance_weight < 0:
                raise ValueError('Empirical goal distance weight cannot be negative.')
            self.empirical_goal_embeddings = goal_embeddings
            self.empirical_goal_distance_weight = float(goal_distance_weight)
        self._plan_one = jax.jit(self._build_plan_one())

    def _build_plan_one(self):
        model = self.model
        variables = self.variables
        num_samples = self.num_samples
        iterations = self.iterations
        topk = self.topk
        var_scale = self.var_scale
        proposal_population_size = self.proposal_population_size
        action_dim = int(self.scaler.action_dim)
        action_block = self.action_block
        planner_action_low = (
            None
            if self.planner_action_low is None
            else jnp.asarray(self.planner_action_low, dtype=jnp.float32)
        )
        planner_action_high = (
            None
            if self.planner_action_high is None
            else jnp.asarray(self.planner_action_high, dtype=jnp.float32)
        )
        temporal_parameterization = self.temporal_parameterization
        empirical_action_blocks = (
            None
            if self.empirical_action_blocks is None
            else jnp.asarray(self.empirical_action_blocks, dtype=jnp.float32)
        )
        empirical_context_embeddings = (
            None
            if self.empirical_context_embeddings is None
            else jnp.asarray(self.empirical_context_embeddings, dtype=jnp.float32)
        )
        empirical_goal_embeddings = (
            None
            if self.empirical_goal_embeddings is None
            else jnp.asarray(self.empirical_goal_embeddings, dtype=jnp.float32)
        )
        empirical_goal_distance_weight = self.empirical_goal_distance_weight
        latent_probe_weight = (
            None
            if self.latent_probe_weight is None
            else jnp.asarray(self.latent_probe_weight, dtype=jnp.float32)
        )
        latent_probe_bias = (
            None
            if self.latent_probe_bias is None
            else jnp.asarray(self.latent_probe_bias, dtype=jnp.float32)
        )
        return_best_candidate = self.return_best_candidate
        empirical_context_rank_penalty = self.empirical_context_rank_penalty
        use_latent_subgoal = self._predict_latent_subgoal is not None

        if self.cost_mode == 'last':
            rollout_cost_method = model.rollout_cost
        elif self.cost_mode == 'moh':
            rollout_cost_method = model.rollout_cost_min_over_horizon
        else:
            raise ValueError(f'Unsupported CEM cost mode: {self.cost_mode!r}.')

        def plan_one(
            key,
            pixels,
            goals,
            target_embedding,
            past_action_blocks,
            initial_mean,
            proposal_blocks,
        ):
            std = jnp.full_like(initial_mean, var_scale)
            nearest_empirical_indices = None
            if empirical_context_embeddings is not None:
                current_embedding = model.apply(
                    variables,
                    pixels[-1][None],
                    train=False,
                    method=model.encode_pixels,
                )[0].astype(jnp.float32)
                context_distances = jnp.sum(
                    (empirical_context_embeddings - current_embedding[None]) ** 2,
                    axis=-1,
                )
                if empirical_goal_embeddings is not None:
                    desired_goal_embedding = model.apply(
                        variables,
                        goals[-1][None],
                        train=False,
                        method=model.encode_pixels,
                    )[0].astype(jnp.float32)
                    goal_distances = jnp.sum(
                        (empirical_goal_embeddings - desired_goal_embedding[None])
                        ** 2,
                        axis=-1,
                    )
                    context_distances = (
                        context_distances
                        + empirical_goal_distance_weight * goal_distances
                    )
                _, nearest_empirical_indices = jax.lax.top_k(
                    -context_distances, num_samples
                )
            def optimizer_step(iteration, carry):
                key, mean, std, best_candidate = carry
                key, sample_key, empirical_key = jax.random.split(key, 3)
                candidates = (
                    jax.random.normal(
                        sample_key,
                        (num_samples, initial_mean.shape[0], initial_mean.shape[1]),
                        dtype=jnp.float32,
                    )
                    * std[None]
                    + mean[None]
                )
                if temporal_parameterization == 'constant':
                    atomic = candidates.reshape(
                        num_samples,
                        initial_mean.shape[0],
                        action_block,
                        action_dim,
                    )
                    candidates = jnp.repeat(
                        atomic[:, :, :1], action_block, axis=2
                    ).reshape(candidates.shape)
                candidates = candidates.at[0].set(mean)
                if proposal_population_size:
                    # The mode remains the exact nominal candidate at index 0;
                    # stochastic actor chunks replace only the first block of
                    # additional iteration-zero candidates.  All candidates
                    # are then ranked solely by the LeWM rollout cost.
                    candidates = jax.lax.cond(
                        iteration == 0,
                        lambda value: value.at[
                            :proposal_population_size, 0
                        ].set(proposal_blocks),
                        lambda value: value,
                        candidates,
                    )
                if empirical_action_blocks is not None:
                    empirical_indices = (
                        nearest_empirical_indices
                        if nearest_empirical_indices is not None
                        else jax.random.randint(
                            empirical_key,
                            (num_samples,),
                            0,
                            empirical_action_blocks.shape[0],
                        )
                    )
                    if empirical_action_blocks.ndim == 2:
                        candidates = jax.lax.cond(
                            iteration == 0,
                            lambda value: value.at[:, 0].set(
                                empirical_action_blocks[empirical_indices]
                            ),
                            lambda value: value,
                            candidates,
                        )
                    else:
                        candidates = jax.lax.cond(
                            iteration == 0,
                            lambda value: value.at[:].set(
                                empirical_action_blocks[empirical_indices]
                            ),
                            lambda value: value,
                            candidates,
                        )
                    if empirical_context_embeddings is None:
                        candidates = candidates.at[0].set(mean)
                if planner_action_low is not None:
                    # Score exactly the bounded actions that the environment can
                    # execute.  Otherwise CEM can exploit predictions for an
                    # out-of-bounds action that is clipped only after planning.
                    candidates = jnp.clip(
                        candidates,
                        planner_action_low[None, None],
                        planner_action_high[None, None],
                    )
                past_candidates = jnp.broadcast_to(
                    past_action_blocks[None],
                    (num_samples, *past_action_blocks.shape),
                )
                scored_candidates = jnp.concatenate(
                    [past_candidates, candidates], axis=1
                )
                if use_latent_subgoal:
                    _, predictions = model.apply(
                        variables,
                        pixels[None, None],
                        goals[None, None],
                        scored_candidates[None],
                        method=model._rollout_predictions,
                    )
                    latent_distances = jnp.sum(
                        (
                            predictions
                            - target_embedding[None, None, None]
                        )
                        ** 2,
                        axis=-1,
                    )
                    costs = select_latent_subgoal_costs(
                        latent_distances,
                        self.cost_mode,
                    )
                elif latent_probe_weight is None:
                    costs = model.apply(
                        variables,
                        pixels[None, None],
                        goals[None, None],
                        scored_candidates[None],
                        method=rollout_cost_method,
                    )[0]
                else:
                    goal_embeddings, predictions = model.apply(
                        variables,
                        pixels[None, None],
                        goals[None, None],
                        scored_candidates[None],
                        method=model._rollout_predictions,
                    )
                    goal_state = (
                        goal_embeddings @ latent_probe_weight + latent_probe_bias
                    )
                    predicted_states = (
                        predictions @ latent_probe_weight + latent_probe_bias
                    )
                    probe_distances = jnp.sum(
                        (
                            predicted_states
                            - goal_state[:, None, None]
                        )
                        ** 2,
                        axis=-1,
                    )
                    if self.cost_mode == 'last':
                        costs = probe_distances[:, :, -1][0]
                    else:
                        costs = jnp.min(probe_distances, axis=-1)[0]
                if empirical_context_embeddings is not None:
                    rank_penalty = (
                        jnp.arange(num_samples, dtype=jnp.float32)
                        / max(num_samples - 1, 1)
                        * empirical_context_rank_penalty
                    )
                    costs = jax.lax.cond(
                        iteration == 0,
                        lambda value: value + rank_penalty,
                        lambda value: value,
                        costs,
                    )
                best_candidate = candidates[jnp.argmin(costs)]
                _, elite_indices = jax.lax.top_k(-costs, topk)
                elites = candidates[elite_indices]
                mean = elites.mean(axis=0)
                std = elites.std(axis=0, ddof=1)
                return key, mean, std, best_candidate

            _, mean, std, best_candidate = jax.lax.fori_loop(
                0,
                iterations,
                optimizer_step,
                (key, initial_mean, std, initial_mean),
            )
            if planner_action_low is not None:
                mean = jnp.clip(mean, planner_action_low, planner_action_high)
                best_candidate = jnp.clip(
                    best_candidate, planner_action_low, planner_action_high
                )
            return (
                best_candidate if return_best_candidate else mean,
                std,
            )

        return plan_one

    def _build_score_plans(self):
        model = self.model
        variables = self.variables
        if self.cost_mode == 'last':
            rollout_cost_method = model.rollout_cost
        elif self.cost_mode == 'moh':
            rollout_cost_method = model.rollout_cost_min_over_horizon
        else:
            raise ValueError(f'Unsupported CEM cost mode: {self.cost_mode!r}.')
        latent_probe_weight = (
            None
            if self.latent_probe_weight is None
            else jnp.asarray(self.latent_probe_weight, dtype=jnp.float32)
        )
        latent_probe_bias = (
            None
            if self.latent_probe_bias is None
            else jnp.asarray(self.latent_probe_bias, dtype=jnp.float32)
        )

        def score_plans(pixels, goals, past_action_blocks, plans):
            past_plans = jnp.broadcast_to(
                past_action_blocks[None],
                (plans.shape[0], *past_action_blocks.shape),
            )
            scored_plans = jnp.concatenate([past_plans, plans], axis=1)
            if latent_probe_weight is None:
                return model.apply(
                    variables,
                    pixels[None, None],
                    goals[None, None],
                    scored_plans[None],
                    method=rollout_cost_method,
                )[0]
            goal_embeddings, predictions = model.apply(
                variables,
                pixels[None, None],
                goals[None, None],
                scored_plans[None],
                method=model._rollout_predictions,
            )
            goal_state = goal_embeddings @ latent_probe_weight + latent_probe_bias
            predicted_states = predictions @ latent_probe_weight + latent_probe_bias
            distances = jnp.sum(
                (predicted_states - goal_state[:, None, None]) ** 2,
                axis=-1,
            )
            if self.cost_mode == 'last':
                return distances[:, :, -1][0]
            return jnp.min(distances, axis=-1)[0]

        return score_plans

    def _build_plan_distances(self):
        model = self.model
        variables = self.variables
        latent_probe_weight = (
            None
            if self.latent_probe_weight is None
            else jnp.asarray(self.latent_probe_weight, dtype=jnp.float32)
        )
        latent_probe_bias = (
            None
            if self.latent_probe_bias is None
            else jnp.asarray(self.latent_probe_bias, dtype=jnp.float32)
        )

        def plan_distances(pixels, goals, past_action_blocks, plan):
            scored_plan = jnp.concatenate([past_action_blocks, plan], axis=0)
            goal_embeddings, predictions = model.apply(
                variables,
                pixels[None, None],
                goals[None, None],
                scored_plan[None, None],
                method=model._rollout_predictions,
            )
            if latent_probe_weight is None:
                return jnp.sum(
                    (predictions[0, 0] - goal_embeddings[0, None]) ** 2,
                    axis=-1,
                )
            goal_state = goal_embeddings @ latent_probe_weight + latent_probe_bias
            predicted_states = predictions @ latent_probe_weight + latent_probe_bias
            return jnp.sum(
                (predicted_states[0, 0] - goal_state[0, None]) ** 2,
                axis=-1,
            )

        return plan_distances

    def reset(self, action_space, num_envs):
        action_dim = int(np.prod(action_space.shape))
        if action_dim != self.scaler.action_dim:
            raise ValueError(f'Environment action dim {action_dim} differs from dataset dim {self.scaler.action_dim}.')
        self.atomic_action_dim = action_dim
        self.block_action_dim = action_dim * self.action_block
        if self.proposal_agent is not None:
            proposal_horizon = int(getattr(self.proposal_agent, 'action_horizon', 1))
            if proposal_horizon != self.action_block:
                raise ValueError(
                    f'Proposal action horizon {proposal_horizon} differs from '
                    f'LeWM action block {self.action_block}.'
                )
        self.buffers = [deque() for _ in range(num_envs)]
        self.warm_starts = [None] * num_envs
        self.pixel_histories = [
            deque(maxlen=self.context_history_size) for _ in range(num_envs)
        ]
        self.action_histories = [
            deque(maxlen=max(self.context_history_size - 1, 1))
            for _ in range(num_envs)
        ]
        self.plan_counts = np.zeros(num_envs, dtype=np.int64)
        self.min_horizon_counts_by_env = np.zeros(
            (num_envs, self.horizon), dtype=np.int64
        )
        self.min_horizon_distance_sums_by_env = np.zeros(
            (num_envs, self.horizon), dtype=np.float64
        )
        self.min_horizon_replans_by_env = np.zeros(num_envs, dtype=np.int64)
        self.latent_subgoal_generation_counts = np.zeros(
            num_envs, dtype=np.int64
        )
        self.latent_subgoal_pixel_histories = [
            deque(maxlen=self.latent_subgoal_history_size) for _ in range(num_envs)
        ]

    def _planning_target_embedding(self, env_index, pixels, goals):
        embed_dim = int(self.checkpoint_metadata['embed_dim'])
        if self._predict_latent_subgoal is None:
            return np.zeros(embed_dim, dtype=np.float32)
        history_size = getattr(self, 'latent_subgoal_history_size', 1)
        if history_size > 1:
            history = list(self.latent_subgoal_pixel_histories[env_index])
            if not history:
                history = [np.asarray(pixels[-1])]
            history = [history[0]] * (history_size - len(history)) + history
            history_embeddings = self.encode_pixels(np.stack(history))
            current_embedding = history_embeddings[None]
        else:
            current_embedding = self.encode_pixels(np.asarray(pixels[-1:]))
        goal_embedding = self.encode_pixels(np.asarray(goals[-1:]))
        if getattr(self, '_latent_subgoal_requires_rng', False):
            subgoal_key = jax.random.fold_in(
                jax.random.PRNGKey(self.seed), int(env_index)
            )
            subgoal_key = jax.random.fold_in(
                subgoal_key,
                int(self.latent_subgoal_generation_counts[env_index]),
            )
            prediction = self._predict_latent_subgoal(
                jnp.asarray(current_embedding),
                jnp.asarray(goal_embedding),
                subgoal_key,
            )
        else:
            prediction = self._predict_latent_subgoal(
                jnp.asarray(current_embedding),
                jnp.asarray(goal_embedding),
            )
        subgoal = np.asarray(prediction)[0].astype(np.float32)
        if subgoal.shape != (embed_dim,) or not np.isfinite(subgoal).all():
            raise FloatingPointError(
                f'Invalid predicted latent subgoal shape/value: {subgoal.shape}.'
            )
        self.latent_subgoal_generation_counts[env_index] += 1
        return subgoal

    def _next_plan_keys(self, env_index):
        """Return proposal/CEM keys, optionally matched across method variants."""
        if self.paired_plan_keys:
            plan_key = jax.random.fold_in(jax.random.PRNGKey(self.seed), int(env_index))
            plan_key = jax.random.fold_in(plan_key, int(self.plan_counts[env_index]))
            self.plan_counts[env_index] += 1
            proposal_key = jax.random.fold_in(plan_key, 1)
            return proposal_key, plan_key

        if self.proposal_agent is None:
            self.rng, plan_key = jax.random.split(self.rng)
            return None, plan_key
        self.rng, proposal_key, plan_key = jax.random.split(self.rng, 3)
        return proposal_key, plan_key

    def _proposal_to_planner_actions(self, blocks):
        """Convert actor outputs into the standardized LeWM planner space."""
        blocks = np.asarray(blocks)
        if self.proposal_action_space == 'planner':
            return blocks
        shape = blocks.shape
        atomic = blocks.reshape(-1, self.atomic_action_dim)
        return self.scaler.transform(atomic).reshape(shape)

    def _selected_proposal_blocks(
        self, pixels, goals, key, past_action_blocks=None
    ):
        """Return actor mode and a LeWM-selected actor sample."""
        sample_key, mode_key = jax.random.split(key)
        observations = np.repeat(
            np.asarray(pixels[-1:]), self.proposal_num_samples, axis=0
        )
        goal_batch = np.repeat(
            np.asarray(goals[-1:]), self.proposal_num_samples, axis=0
        )
        proposal_blocks = np.asarray(
            self.proposal_agent.sample_actions(
                observations=observations,
                goals=goal_batch,
                seed=sample_key,
                temperature=self.proposal_temperature,
            )
        ).copy()
        mode = np.asarray(
            self.proposal_agent.sample_actions(
                observations=np.asarray(pixels[-1:]),
                goals=np.asarray(goals[-1:]),
                seed=mode_key,
                temperature=0.0,
            )
        )[0]
        if proposal_blocks.shape != (
            self.proposal_num_samples,
            self.block_action_dim,
        ):
            raise ValueError(
                f'Proposal returned {proposal_blocks.shape}; expected '
                f'({self.proposal_num_samples}, {self.block_action_dim}).'
            )
        proposal_blocks[0] = mode
        planner_blocks = self._proposal_to_planner_actions(proposal_blocks)
        if self.proposal_selection in ('lewm', 'lewm_cem'):
            plans = np.zeros(
                (
                    self.proposal_num_samples,
                    self.horizon,
                    self.block_action_dim,
                ),
                dtype=np.float32,
            )
            plans[:, 0] = planner_blocks
            if past_action_blocks is None:
                costs = np.asarray(
                    self._score_plans(
                        jnp.asarray(pixels),
                        jnp.asarray(goals),
                        jnp.asarray(plans),
                    )
                )
            else:
                costs = np.asarray(
                    self._score_plans(
                        jnp.asarray(pixels),
                        jnp.asarray(goals),
                        jnp.asarray(past_action_blocks),
                        jnp.asarray(plans),
                    )
                )
            if self.proposal_selection == 'lewm':
                selected_block = planner_blocks[int(np.argmin(costs))]
            else:
                elite_indices = np.argsort(costs)[: self.proposal_elite_size]
                elite_mean = planner_blocks[elite_indices].mean(axis=0)
                selected_block = planner_blocks[0] + self.proposal_residual_weight * (
                    elite_mean - planner_blocks[0]
                )
        else:
            raise ValueError(
                f'Unsupported proposal selection: {self.proposal_selection!r}.'
            )
        return (
            planner_blocks[0],
            selected_block,
        )

    def _proposal_block(
        self, pixels, goals, key, past_action_blocks=None
    ):
        if self.proposal_selection == 'mode':
            proposal_block = np.asarray(
                self.proposal_agent.sample_actions(
                    observations=np.asarray(pixels[-1:]),
                    goals=np.asarray(goals[-1:]),
                    seed=key,
                    temperature=self.proposal_temperature,
                )
            )
            if proposal_block.shape != (1, self.block_action_dim):
                raise ValueError(
                    f'Proposal returned {proposal_block.shape}; expected '
                    f'(1, {self.block_action_dim}).'
                )
            return self._proposal_to_planner_actions(proposal_block[0])

        _, selected = self._selected_proposal_blocks(
            pixels, goals, key, past_action_blocks
        )
        return selected

    def _proposal_population(self, pixels, goals, key):
        sample_key, mode_key = jax.random.split(key)
        count = self.proposal_population_size
        observations = np.repeat(np.asarray(pixels[-1:]), count, axis=0)
        goal_batch = np.repeat(np.asarray(goals[-1:]), count, axis=0)
        blocks = np.asarray(
            self.proposal_agent.sample_actions(
                observations=observations,
                goals=goal_batch,
                seed=sample_key,
                temperature=self.proposal_temperature,
            )
        ).copy()
        if blocks.shape != (count, self.block_action_dim):
            raise ValueError(
                f'Proposal population returned {blocks.shape}; expected '
                f'({count}, {self.block_action_dim}).'
            )
        blocks[0] = np.asarray(
            self.proposal_agent.sample_actions(
                observations=np.asarray(pixels[-1:]),
                goals=np.asarray(goals[-1:]),
                seed=mode_key,
                temperature=0.0,
            )
        )[0]
        return self._proposal_to_planner_actions(blocks)

    def _initial_mean(
        self,
        env_index,
        pixels=None,
        goals=None,
        proposal_key=None,
        proposal_block=None,
        past_action_blocks=None,
    ):
        initial = np.zeros((self.horizon, self.block_action_dim), dtype=np.float32)
        warm = self.warm_starts[env_index]
        if warm is not None:
            initial[: len(warm)] = warm
        if self.proposal_agent is not None:
            if proposal_block is not None:
                initial[0] = proposal_block
            elif pixels is None or goals is None or proposal_key is None:
                raise ValueError('Proposal-guided CEM requires pixels, goals, and a PRNG key.')
            else:
                # GCIQL-Chunk predicts exactly one normalized action block from
                # the real current image and dataset goal.  Only replace the
                # first planner block; the remaining horizon stays under the
                # original LeWM warm start / zero initialization.
                initial[0] = self._proposal_block(
                    pixels, goals, proposal_key, past_action_blocks
                )
        return initial

    def _planning_context(self, env_index, pixels):
        current = np.asarray(pixels[-1])
        pixel_history = self.pixel_histories[env_index]
        pixel_history.append(current)
        context = list(pixel_history)
        context = [context[0]] * (self.context_history_size - len(context)) + context

        past = list(self.action_histories[env_index])
        missing = self.context_history_size - 1 - len(past)
        past = [np.zeros(self.block_action_dim, dtype=np.float32)] * missing + past
        return np.stack(context), np.asarray(past, dtype=np.float32).reshape(
            self.context_history_size - 1, self.block_action_dim
        )

    def get_actions(self, pixels, goals, alive):
        for env_index in np.flatnonzero(alive):
            if (
                self._predict_latent_subgoal is not None
                and self.latent_subgoal_history_size > 1
            ):
                self.latent_subgoal_pixel_histories[env_index].append(
                    np.asarray(pixels[env_index][-1])
                )
            if self.buffers[env_index]:
                continue
            context_pixels, past_action_blocks = self._planning_context(
                env_index, pixels[env_index]
            )
            proposal_key, plan_key = self._next_plan_keys(env_index)
            target_embedding = self._planning_target_embedding(
                env_index, context_pixels, goals[env_index]
            )
            if self.proposal_population_size:
                proposal_blocks = self._proposal_population(
                    pixels[env_index], goals[env_index], proposal_key
                )
                initial_mean = self._initial_mean(
                    env_index, proposal_block=proposal_blocks[0]
                )
            else:
                initial_mean = self._initial_mean(
                    env_index,
                    pixels=context_pixels,
                    goals=goals[env_index],
                    proposal_key=proposal_key,
                    past_action_blocks=past_action_blocks,
                )
                proposal_blocks = np.zeros(
                    (0, self.block_action_dim), dtype=np.float32
                )
            normalized_blocks, _ = self._plan_one(
                plan_key,
                jnp.asarray(context_pixels),
                jnp.asarray(goals[env_index]),
                jnp.asarray(target_embedding),
                jnp.asarray(past_action_blocks),
                jnp.asarray(initial_mean),
                jnp.asarray(proposal_blocks),
            )
            normalized_blocks = np.asarray(normalized_blocks)
            if self.diagnose_min_horizon:
                distances = np.asarray(
                    self._plan_distances(
                        jnp.asarray(context_pixels),
                        jnp.asarray(goals[env_index]),
                        jnp.asarray(past_action_blocks),
                        jnp.asarray(normalized_blocks),
                    )
                )
                best_horizon = int(np.argmin(distances))
                self.min_horizon_counts_by_env[env_index, best_horizon] += 1
                self.min_horizon_distance_sums_by_env[env_index] += distances
                self.min_horizon_replans_by_env[env_index] += 1
            keep = normalized_blocks[: self.receding_horizon]
            self.warm_starts[env_index] = normalized_blocks[
                self.receding_horizon:
            ].copy()
            normalized_atomic = keep.reshape(-1, self.atomic_action_dim)
            if self.context_history_size > 1:
                self.action_histories[env_index].append(normalized_blocks[0].copy())
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
    if (args.proposal_method is None) != (args.proposal_checkpoint_dir is None):
        raise ValueError(
            '--proposal-method and --proposal-checkpoint-dir must be provided together.'
        )

    hdf5_path, lance_path = task_paths(args.task, args.data_root)
    dataset = HDF5EvaluationDataset(hdf5_path)
    try:
        episodes, starts = dataset.sample_starts(args.num_eval, args.goal_offset_steps, args.seed)
        scaler = StandardActionScaler(dataset.get_column('action'))
        proposal_agent = None
        if args.proposal_method is not None:
            from gciql_chunk_policy import load_lance_policy

            proposal_agent = load_lance_policy(
                lance_path,
                args.proposal_checkpoint_dir,
                args.proposal_checkpoint_step,
            )
        policy = JAXLeWMCEMPolicy(
            args.checkpoint,
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
            proposal_agent=proposal_agent,
            proposal_temperature=args.proposal_temperature,
            proposal_action_space='planner',
            proposal_num_samples=args.proposal_num_samples,
            proposal_population_size=args.proposal_population_size,
            proposal_selection=args.proposal_selection,
            proposal_elite_size=args.proposal_elite_size,
            proposal_residual_weight=args.proposal_residual_weight,
            paired_plan_keys=args.paired_plan_keys,
            diagnose_min_horizon=args.diagnose_min_horizon,
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

    min_horizon_diagnostics = None
    if args.diagnose_min_horizon:
        episode_successes = np.asarray(metrics['episode_successes'], dtype=bool)
        total_counts = policy.min_horizon_counts_by_env.sum(axis=0)
        successful_counts = policy.min_horizon_counts_by_env[episode_successes].sum(axis=0)
        failed_counts = policy.min_horizon_counts_by_env[~episode_successes].sum(axis=0)
        total_replans = int(policy.min_horizon_replans_by_env.sum())
        distance_sums = policy.min_horizon_distance_sums_by_env.sum(axis=0)
        min_horizon_diagnostics = {
            'checkpoint_atomic_steps': [
                args.action_block * (index + 1) for index in range(args.cem_horizon)
            ],
            'argmin_counts': total_counts,
            'argmin_percentages': (
                total_counts / max(total_replans, 1) * 100.0
            ),
            'successful_episode_argmin_counts': successful_counts,
            'failed_episode_argmin_counts': failed_counts,
            'mean_selected_plan_distance_by_horizon': (
                distance_sums / max(total_replans, 1)
            ),
            'total_replans': total_replans,
            'replans_by_episode': policy.min_horizon_replans_by_env,
            'argmin_counts_by_episode': policy.min_horizon_counts_by_env,
        }

    result = {
        'task': args.task,
        'method': (
            'lewm_jax_cem'
            if args.proposal_method is None
            else f'lewm_jax_cem_{args.proposal_method}_proposal'
        ),
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
            'name': 'cem',
            'requested_horizon': args.cem_horizon,
            'horizon': policy.horizon,
            'receding_horizon': args.cem_receding_horizon,
            'action_block': args.action_block,
            'num_samples': args.cem_num_samples,
            'iterations': args.cem_iterations,
            'topk': args.cem_topk,
            'var_scale': args.cem_var_scale,
            'cost_mode': args.cem_cost_mode,
            'batch_size': 1,
            'history_len': 1,
            'warm_start': True,
            'paired_plan_keys': args.paired_plan_keys,
        },
        'proposal': (
            None
            if args.proposal_method is None
            else {
                'method': args.proposal_method,
                'checkpoint_dir': args.proposal_checkpoint_dir,
                'checkpoint_step': args.proposal_checkpoint_step,
                'temperature': args.proposal_temperature,
                'action_space': 'planner',
                'injection': 'first_block_initial_mean',
                'num_samples': args.proposal_num_samples,
                'population_injection_size': args.proposal_population_size,
                'population_injection_scope': (
                    'cem_iteration_zero_first_block_lewm_only_ranking'
                    if args.proposal_population_size > 0
                    else 'disabled'
                ),
                'selection': args.proposal_selection,
                'elite_size': args.proposal_elite_size,
                'residual_weight': args.proposal_residual_weight,
            }
        ),
        'eval_episodes': episodes,
        'eval_start_steps': starts,
        'evaluation_time': elapsed,
        'metrics': metrics,
        'min_horizon_diagnostics': min_horizon_diagnostics,
        'success_rate': metrics['success_rate'],
        'episodes': args.num_eval,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(json_safe(result), indent=2) + '\n')
    print(json.dumps(json_safe(result), indent=2))


if __name__ == '__main__':
    main()
