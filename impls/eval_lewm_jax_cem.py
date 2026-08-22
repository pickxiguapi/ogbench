"""Evaluate a JAX LeWM checkpoint with CEM or MPPI in LeWM environments."""

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
    parser.add_argument('--planner', choices=('cem', 'mppi'), default='cem')
    parser.add_argument('--cem-horizon', type=int, default=5)
    parser.add_argument('--cem-receding-horizon', type=int, default=5)
    parser.add_argument('--action-block', type=int, default=5)
    parser.add_argument('--cem-num-samples', type=int, default=300)
    parser.add_argument('--cem-steps', type=int, default=30)
    parser.add_argument('--cem-topk', type=int, default=30)
    parser.add_argument('--cem-var-scale', type=float, default=1.0)
    parser.add_argument(
        '--mppi-temperature',
        type=float,
        default=0.5,
        help='Temperature for softmax weighting of MPPI top-k candidates.',
    )
    parser.add_argument(
        '--mppi-native-q-beta',
        type=float,
        default=0.0,
        help=(
            'If positive, add beta times standardized conservative native-Q '
            'to the MPPI logits after LeWM has selected its top-k candidates.'
        ),
    )
    parser.add_argument(
        '--cem-cost-mode',
        choices=('terminal', 'min_over_horizon'),
        default='terminal',
    )
    parser.add_argument('--proposal-method', choices=('gciql_chunk',))
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
        choices=('mode', 'lewm', 'lewm_cem', 'native_q', 'shared_q'),
        default='mode',
    )
    parser.add_argument('--proposal-elite-size', type=int, default=1)
    parser.add_argument('--proposal-residual-weight', type=float, default=1.0)
    parser.add_argument(
        '--cem-dual-center-q',
        action='store_true',
        help=(
            'Run two independent half-budget CEM searches centered on the '
            'actor mode and Q-selected actor sample, then let LeWM cost choose '
            'between their final means. The total population remains '
            '--cem-num-samples and top-k is split evenly.'
        ),
    )
    parser.add_argument(
        '--native-q-keep',
        type=int,
        default=0,
        help=(
            'If positive, keep this many candidates by the proposal agent twin-Q '
            'before selecting LeWM elites at every CEM iteration.'
        ),
    )
    parser.add_argument('--shared-q-checkpoint-dir')
    parser.add_argument('--shared-q-checkpoint-step', type=int, default=100_000)
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


class JAXLeWMCEMPolicy:
    """Direct JAX CEM/MPPI planner with normalized action-block warm starts."""

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
        planner='cem',
        mppi_temperature=0.5,
        mppi_native_q_beta=0.0,
        cost_mode='terminal',
        proposal_agent=None,
        proposal_temperature=0.0,
        proposal_action_space='planner',
        proposal_num_samples=1,
        proposal_population_size=0,
        proposal_selection='mode',
        proposal_elite_size=1,
        proposal_residual_weight=1.0,
        dual_center_q=False,
        native_q_keep=0,
        shared_q_evaluator=None,
        paired_plan_keys=False,
        diagnose_min_horizon=False,
        execution_steps=None,
        action_low=None,
        action_high=None,
        temporal_parameterization='independent',
        empirical_action_blocks=None,
    ):
        if horizon <= 0 or receding_horizon <= 0 or action_block <= 0:
            raise ValueError('CEM horizon, receding horizon, and action block must be positive.')
        if receding_horizon > horizon:
            raise ValueError('CEM receding horizon cannot exceed the planning horizon.')
        if execution_steps is not None and not (
            1 <= execution_steps <= receding_horizon * action_block
        ):
            raise ValueError(
                'Execution steps must be in [1, receding_horizon * action_block].'
            )
        if (action_low is None) != (action_high is None):
            raise ValueError('Action low and high bounds must be provided together.')
        if temporal_parameterization not in ('independent', 'constant'):
            raise ValueError(
                'Temporal action parameterization must be independent or constant.'
            )
        if not 1 < topk <= num_samples:
            raise ValueError('CEM topk must be in [2, num_samples].')
        if planner not in ('cem', 'mppi'):
            raise ValueError(f'Unsupported planner: {planner!r}.')
        if mppi_temperature <= 0:
            raise ValueError('MPPI temperature must be positive.')
        if mppi_native_q_beta < 0:
            raise ValueError('MPPI native-Q beta cannot be negative.')
        if mppi_native_q_beta and planner != 'mppi':
            raise ValueError('MPPI native-Q weighting requires planner=mppi.')
        if mppi_native_q_beta and proposal_agent is None:
            raise ValueError('MPPI native-Q weighting requires a proposal agent.')
        if planner == 'mppi' and native_q_keep:
            raise ValueError('Native-Q candidate filtering is only defined for CEM.')
        if native_q_keep < 0 or native_q_keep > num_samples:
            raise ValueError('Native-Q keep count must be in [0, num_samples].')
        if native_q_keep and proposal_agent is None:
            raise ValueError('Native-Q filtering requires a proposal agent.')
        if native_q_keep and native_q_keep < topk:
            raise ValueError('Native-Q keep count cannot be smaller than CEM topk.')
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
        if proposal_selection == 'shared_q' and shared_q_evaluator is None:
            raise ValueError('Shared-Q proposal selection requires a shared evaluator.')
        if dual_center_q:
            if planner != 'cem':
                raise ValueError('Dual-center Q search requires planner=cem.')
            if proposal_selection not in ('native_q', 'shared_q'):
                raise ValueError('Dual-center Q search requires Q proposal selection.')
            if proposal_agent is None:
                raise ValueError('Dual-center Q search requires a proposal agent.')
            if proposal_population_size or native_q_keep:
                raise ValueError(
                    'Dual-center Q search cannot be combined with population '
                    'injection or per-iteration Q filtering.'
                )
            if num_samples % 2 or topk % 2:
                raise ValueError(
                    'Dual-center Q search requires even num-samples and top-k.'
                )
            if topk // 2 < 2:
                raise ValueError('Each dual-center search needs at least two elites.')

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
            'embed_dim': int(config['embed_dim']),
        }
        if shared_q_evaluator is not None and int(
            shared_q_evaluator.config['latent_dim']
        ) != int(config['embed_dim']):
            raise ValueError('Shared evaluator and LeWM latent dimensions differ.')
        self.scaler = scaler
        self.seed = int(seed)
        self.rng = jax.random.PRNGKey(seed)
        self.horizon = int(horizon)
        self.receding_horizon = int(receding_horizon)
        self.action_block = int(action_block)
        self.temporal_parameterization = str(temporal_parameterization)
        self.empirical_action_blocks = None
        if empirical_action_blocks is not None:
            empirical_action_blocks = np.asarray(
                empirical_action_blocks, dtype=np.float32
            )
            expected_width = int(self.scaler.action_dim) * self.action_block
            if (
                empirical_action_blocks.ndim != 2
                or empirical_action_blocks.shape[1] != expected_width
                or empirical_action_blocks.shape[0] < 2
            ):
                raise ValueError(
                    'Empirical action blocks must have shape '
                    f'(N >= 2, {expected_width}); got '
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
        self.steps = int(steps)
        self.topk = int(topk)
        self.var_scale = float(var_scale)
        self.planner = str(planner)
        self.mppi_temperature = float(mppi_temperature)
        self.mppi_native_q_beta = float(mppi_native_q_beta)
        self.cost_mode = str(cost_mode)
        self.proposal_agent = proposal_agent
        self.proposal_temperature = float(proposal_temperature)
        self.proposal_action_space = str(proposal_action_space)
        self.proposal_num_samples = int(proposal_num_samples)
        self.proposal_population_size = int(proposal_population_size)
        self.proposal_selection = str(proposal_selection)
        self.proposal_elite_size = int(proposal_elite_size)
        self.proposal_residual_weight = float(proposal_residual_weight)
        self.dual_center_q = bool(dual_center_q)
        self.native_q_keep = int(native_q_keep)
        self.shared_q_evaluator = shared_q_evaluator
        self.paired_plan_keys = bool(paired_plan_keys)
        self.diagnose_min_horizon = bool(diagnose_min_horizon)
        self.execution_steps = (
            self.receding_horizon * self.action_block
            if execution_steps is None
            else int(execution_steps)
        )
        self._plan_one = jax.jit(self._build_plan_one())
        self._score_plans = jax.jit(self._build_score_plans())
        self._plan_distances = jax.jit(self._build_plan_distances())

    def _build_plan_one(self):
        model = self.model
        variables = self.variables
        num_samples = self.num_samples // 2 if self.dual_center_q else self.num_samples
        steps = self.steps
        topk = self.topk // 2 if self.dual_center_q else self.topk
        var_scale = self.var_scale
        planner = self.planner
        mppi_temperature = self.mppi_temperature
        mppi_native_q_beta = self.mppi_native_q_beta
        proposal_agent = self.proposal_agent
        proposal_action_space = self.proposal_action_space
        native_q_keep = self.native_q_keep
        shared_q_evaluator = self.shared_q_evaluator
        proposal_population_size = self.proposal_population_size
        action_dim = int(self.scaler.action_dim)
        action_block = self.action_block
        action_mean = jnp.asarray(self.scaler.mean, dtype=jnp.float32)
        action_scale = jnp.asarray(self.scaler.scale, dtype=jnp.float32)
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

        def planner_to_proposal_actions(blocks):
            if proposal_action_space == 'planner':
                return blocks
            shape = blocks.shape
            atomic = blocks.reshape(-1, action_block, action_dim)
            atomic = atomic * action_scale + action_mean
            return atomic.reshape(shape)
        if self.cost_mode == 'terminal':
            rollout_cost_method = model.rollout_cost
        elif self.cost_mode == 'min_over_horizon':
            rollout_cost_method = model.rollout_cost_min_over_horizon
        else:
            raise ValueError(f'Unsupported CEM cost mode: {self.cost_mode!r}.')

        def plan_one(key, pixels, goals, initial_mean, proposal_blocks):
            std = jnp.full_like(initial_mean, var_scale)
            if native_q_keep:
                if shared_q_evaluator is None:
                    q_observations = jnp.repeat(
                        pixels[-1][None], num_samples, axis=0
                    )
                    q_goals = jnp.repeat(goals[-1][None], num_samples, axis=0)
                else:
                    observation_latent = model.apply(
                        variables,
                        pixels[-1][None],
                        train=False,
                        method=model.encode_pixels,
                    ).astype(jnp.float32)
                    goal_latent = model.apply(
                        variables,
                        goals[-1][None],
                        train=False,
                        method=model.encode_pixels,
                    ).astype(jnp.float32)
                    q_observations = jnp.repeat(
                        observation_latent, num_samples, axis=0
                    )
                    q_goals = jnp.repeat(goal_latent, num_samples, axis=0)

            def optimizer_step(iteration, carry):
                key, mean, std = carry
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
                    empirical_indices = jax.random.randint(
                        empirical_key,
                        (num_samples,),
                        0,
                        empirical_action_blocks.shape[0],
                    )
                    candidates = jax.lax.cond(
                        iteration == 0,
                        lambda value: value.at[:, 0].set(
                            empirical_action_blocks[empirical_indices]
                        ),
                        lambda value: value,
                        candidates,
                    )
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
                costs = model.apply(
                    variables,
                    pixels[None, None],
                    goals[None, None],
                    candidates[None],
                    method=rollout_cost_method,
                )[0]
                if native_q_keep:
                    q_action_blocks = planner_to_proposal_actions(candidates[:, 0])
                    if shared_q_evaluator is None:
                        q1, q2 = proposal_agent.network.select('critic')(
                            q_observations,
                            q_goals,
                            q_action_blocks,
                        )
                        q_values = jnp.minimum(q1, q2)
                    else:
                        q_values = shared_q_evaluator.score_actions(
                            q_observations, q_goals, q_action_blocks
                        )
                    _, q_indices = jax.lax.top_k(q_values, native_q_keep)
                    gated_costs = jnp.full_like(costs, jnp.inf)
                    costs = gated_costs.at[q_indices].set(costs[q_indices])
                if planner == 'cem':
                    _, elite_indices = jax.lax.top_k(-costs, topk)
                    elites = candidates[elite_indices]
                    mean = elites.mean(axis=0)
                    std = elites.std(axis=0, ddof=1)
                else:
                    # Match stable-worldmodel's official MPPISolver: select
                    # top-k, softmax raw shifted costs, update only the mean,
                    # and keep the sampling scale fixed across iterations.
                    relevant_costs, relevant_indices = jax.lax.top_k(
                        -costs, topk
                    )
                    relevant_costs = -relevant_costs
                    relevant_candidates = candidates[relevant_indices]
                    logits = -(
                        relevant_costs - jnp.min(relevant_costs)
                    ) / mppi_temperature
                    if mppi_native_q_beta:
                        q_observations = jnp.repeat(
                            pixels[-1][None], topk, axis=0
                        )
                        q_goals = jnp.repeat(goals[-1][None], topk, axis=0)
                        q_action_blocks = planner_to_proposal_actions(
                            relevant_candidates[:, 0]
                        )
                        q1, q2 = proposal_agent.network.select('critic')(
                            q_observations,
                            q_goals,
                            q_action_blocks,
                        )
                        q_values = jnp.minimum(q1, q2)
                        normalized_q = (
                            q_values - jnp.mean(q_values)
                        ) / jnp.maximum(jnp.std(q_values), 1e-6)
                        normalized_q = jnp.clip(normalized_q, -3.0, 3.0)
                        logits = logits + mppi_native_q_beta * normalized_q
                    weights = jax.nn.softmax(logits)
                    mean = jnp.sum(
                        weights[:, None, None] * relevant_candidates, axis=0
                    )
                return key, mean, std

            _, mean, std = jax.lax.fori_loop(
                0, steps, optimizer_step, (key, initial_mean, std)
            )
            if planner_action_low is not None:
                mean = jnp.clip(mean, planner_action_low, planner_action_high)
            return mean, std

        return plan_one

    def _build_score_plans(self):
        model = self.model
        variables = self.variables
        if self.cost_mode == 'terminal':
            rollout_cost_method = model.rollout_cost
        elif self.cost_mode == 'min_over_horizon':
            rollout_cost_method = model.rollout_cost_min_over_horizon
        else:
            raise ValueError(f'Unsupported CEM cost mode: {self.cost_mode!r}.')

        def score_plans(pixels, goals, plans):
            return model.apply(
                variables,
                pixels[None, None],
                goals[None, None],
                plans[None],
                method=rollout_cost_method,
            )[0]

        return score_plans

    def _build_plan_distances(self):
        model = self.model
        variables = self.variables

        def plan_distances(pixels, goals, plan):
            goal_embeddings, predictions = model.apply(
                variables,
                pixels[None, None],
                goals[None, None],
                plan[None, None],
                method=model._rollout_predictions,
            )
            return jnp.sum(
                (predictions[0, 0] - goal_embeddings[0, None]) ** 2,
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
        self.plan_counts = np.zeros(num_envs, dtype=np.int64)
        self.dual_center_choice_counts = np.zeros(2, dtype=np.int64)
        self.min_horizon_counts_by_env = np.zeros(
            (num_envs, self.horizon), dtype=np.int64
        )
        self.min_horizon_distance_sums_by_env = np.zeros(
            (num_envs, self.horizon), dtype=np.float64
        )
        self.min_horizon_replans_by_env = np.zeros(num_envs, dtype=np.int64)

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

    def _q_selection_blocks(self, pixels, goals, key):
        """Return actor mode and a LeWM/Q-selected actor sample."""
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
            costs = np.asarray(
                self._score_plans(
                    jnp.asarray(pixels),
                    jnp.asarray(goals),
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
        elif self.proposal_selection == 'native_q':
            q1, q2 = self.proposal_agent.network.select('critic')(
                observations, goal_batch, jnp.asarray(proposal_blocks)
            )
            q_values = jnp.minimum(q1, q2)
            selected_block = planner_blocks[int(jnp.argmax(q_values))]
        else:
            observation_latent = self.model.apply(
                self.variables,
                jnp.asarray(pixels[-1:]),
                train=False,
                method=self.model.encode_pixels,
            ).astype(jnp.float32)
            goal_latent = self.model.apply(
                self.variables,
                jnp.asarray(goals[-1:]),
                train=False,
                method=self.model.encode_pixels,
            ).astype(jnp.float32)
            q_values = self.shared_q_evaluator.score_actions(
                jnp.repeat(observation_latent, self.proposal_num_samples, axis=0),
                jnp.repeat(goal_latent, self.proposal_num_samples, axis=0),
                jnp.asarray(proposal_blocks),
            )
            selected_block = planner_blocks[int(jnp.argmax(q_values))]
        return (
            planner_blocks[0],
            selected_block,
        )

    def _proposal_block(self, pixels, goals, key):
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

        _, q_selected = self._q_selection_blocks(pixels, goals, key)
        return q_selected

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
                initial[0] = self._proposal_block(pixels, goals, proposal_key)
        return initial

    def get_actions(self, pixels, goals, alive):
        for env_index in np.flatnonzero(alive):
            if self.buffers[env_index]:
                continue
            proposal_key, plan_key = self._next_plan_keys(env_index)
            proposal_blocks = None
            if self.dual_center_q:
                mode_block, q_block = self._q_selection_blocks(
                    pixels[env_index], goals[env_index], proposal_key
                )
                mode_initial = self._initial_mean(
                    env_index, proposal_block=mode_block
                )
                q_initial = self._initial_mean(
                    env_index, proposal_block=q_block
                )
                empty_blocks = jnp.zeros(
                    (0, self.block_action_dim), dtype=jnp.float32
                )
                mode_plan, _ = self._plan_one(
                    jax.random.fold_in(plan_key, 2),
                    jnp.asarray(pixels[env_index]),
                    jnp.asarray(goals[env_index]),
                    jnp.asarray(mode_initial),
                    empty_blocks,
                )
                q_plan, _ = self._plan_one(
                    jax.random.fold_in(plan_key, 3),
                    jnp.asarray(pixels[env_index]),
                    jnp.asarray(goals[env_index]),
                    jnp.asarray(q_initial),
                    empty_blocks,
                )
                candidate_plans = jnp.stack((mode_plan, q_plan))
                plan_costs = self._score_plans(
                    jnp.asarray(pixels[env_index]),
                    jnp.asarray(goals[env_index]),
                    candidate_plans,
                )
                chosen_center = int(jnp.argmin(plan_costs))
                self.dual_center_choice_counts[chosen_center] += 1
                normalized_blocks = np.asarray(candidate_plans[chosen_center])
            elif self.proposal_population_size:
                proposal_blocks = self._proposal_population(
                    pixels[env_index], goals[env_index], proposal_key
                )
                initial_mean = self._initial_mean(
                    env_index, proposal_block=proposal_blocks[0]
                )
            else:
                initial_mean = self._initial_mean(
                    env_index,
                    pixels=pixels[env_index],
                    goals=goals[env_index],
                    proposal_key=proposal_key,
                )
                proposal_blocks = np.zeros(
                    (0, self.block_action_dim), dtype=np.float32
                )
            if not self.dual_center_q:
                normalized_blocks, _ = self._plan_one(
                    plan_key,
                    jnp.asarray(pixels[env_index]),
                    jnp.asarray(goals[env_index]),
                    jnp.asarray(initial_mean),
                    jnp.asarray(proposal_blocks),
                )
                normalized_blocks = np.asarray(normalized_blocks)
            if self.diagnose_min_horizon:
                distances = np.asarray(
                    self._plan_distances(
                        jnp.asarray(pixels[env_index]),
                        jnp.asarray(goals[env_index]),
                        jnp.asarray(normalized_blocks),
                    )
                )
                best_horizon = int(np.argmin(distances))
                self.min_horizon_counts_by_env[env_index, best_horizon] += 1
                self.min_horizon_distance_sums_by_env[env_index] += distances
                self.min_horizon_replans_by_env[env_index] += 1
            keep = normalized_blocks[: self.receding_horizon]
            if self.execution_steps % self.action_block == 0:
                executed_blocks = self.execution_steps // self.action_block
                self.warm_starts[env_index] = normalized_blocks[
                    executed_blocks:
                ].copy()
            else:
                # A partially executed action block no longer aligns with the
                # blockwise warm start expected by LeWM, so replan from scratch.
                self.warm_starts[env_index] = None
            normalized_atomic = keep.reshape(-1, self.atomic_action_dim)
            normalized_atomic = normalized_atomic[: self.execution_steps]
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


def load_shared_q_evaluator(checkpoint_dir, checkpoint_step, action_width):
    """Restore latent Q/V heads trained by train_lewm_with_gciql_chunk.py."""
    from agents.gciql_chunk_lewm_shared import (
        LeWMSharedGCIQLChunkEvaluator,
        get_config,
    )
    from utils.flax_utils import restore_agent

    flags_path = Path(checkpoint_dir) / 'flags.json'
    if not flags_path.is_file():
        raise FileNotFoundError(f'Shared evaluator flags not found: {flags_path}')
    saved = json.loads(flags_path.read_text())
    config = get_config()
    for key, value in saved.get('agent', {}).items():
        if key in config:
            config[key] = value
    latent_dim = int(config.latent_dim)
    evaluator = LeWMSharedGCIQLChunkEvaluator.create(
        0,
        jnp.zeros((1, latent_dim), dtype=jnp.float32),
        jnp.zeros((1, action_width), dtype=jnp.float32),
        config,
    )
    return restore_agent(evaluator, checkpoint_dir, checkpoint_step)


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
            from eval_ogbench_agent_lewm_envs import load_agent

            proposal_agent = load_agent(
                args.proposal_method,
                lance_path,
                args.proposal_checkpoint_dir,
                args.proposal_checkpoint_step,
            )
        shared_q_evaluator = None
        if args.shared_q_checkpoint_dir is not None:
            if args.native_q_keep <= 0:
                raise ValueError('Shared-Q checkpoint requires --native-q-keep > 0.')
            shared_q_evaluator = load_shared_q_evaluator(
                args.shared_q_checkpoint_dir,
                args.shared_q_checkpoint_step,
                scaler.action_dim * args.action_block,
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
            planner=args.planner,
            mppi_temperature=args.mppi_temperature,
            mppi_native_q_beta=args.mppi_native_q_beta,
            cost_mode=args.cem_cost_mode,
            proposal_agent=proposal_agent,
            proposal_temperature=args.proposal_temperature,
            proposal_action_space='planner',
            proposal_num_samples=args.proposal_num_samples,
            proposal_population_size=args.proposal_population_size,
            proposal_selection=args.proposal_selection,
            proposal_elite_size=args.proposal_elite_size,
            proposal_residual_weight=args.proposal_residual_weight,
            dual_center_q=args.cem_dual_center_q,
            native_q_keep=args.native_q_keep,
            shared_q_evaluator=shared_q_evaluator,
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
            f'lewm_jax_{args.planner}'
            if args.proposal_method is None
            else f'lewm_jax_{args.planner}_{args.proposal_method}_proposal'
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
        args.planner: {
            'name': args.planner,
            'horizon': args.cem_horizon,
            'receding_horizon': args.cem_receding_horizon,
            'action_block': args.action_block,
            'num_samples': args.cem_num_samples,
            'steps': args.cem_steps,
            'topk': args.cem_topk,
            'var_scale': args.cem_var_scale,
            'mppi_temperature': (
                args.mppi_temperature if args.planner == 'mppi' else None
            ),
            'mppi_native_q_beta': (
                args.mppi_native_q_beta if args.planner == 'mppi' else None
            ),
            'mppi_native_q_scope': (
                'lewm_topk_first_block_each_iteration'
                if args.mppi_native_q_beta > 0
                else 'disabled'
            ),
            'mppi_official_stable_worldmodel_update': args.planner == 'mppi',
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
                'dual_center_q': args.cem_dual_center_q,
                'dual_center_total_population': (
                    args.cem_num_samples if args.cem_dual_center_q else None
                ),
                'dual_center_population_per_branch': (
                    args.cem_num_samples // 2 if args.cem_dual_center_q else None
                ),
                'dual_center_topk_per_branch': (
                    args.cem_topk // 2 if args.cem_dual_center_q else None
                ),
                'dual_center_choice_counts_mode_q': (
                    policy.dual_center_choice_counts.tolist()
                    if args.cem_dual_center_q
                    else None
                ),
                'native_q_keep': args.native_q_keep,
                'native_q_scope': (
                    'disabled' if args.native_q_keep == 0 else 'each_cem_iteration_first_block'
                ),
                'q_source': (
                    'native_gciql'
                    if args.shared_q_checkpoint_dir is None
                    else 'shared_frozen_lewm_encoder'
                ),
                'shared_q_checkpoint_dir': args.shared_q_checkpoint_dir,
                'shared_q_checkpoint_step': (
                    None
                    if args.shared_q_checkpoint_dir is None
                    else args.shared_q_checkpoint_step
                ),
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
