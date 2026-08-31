"""Checkpoint loading and execution adapters for final GCIQL-Chunk policies."""

from __future__ import annotations

import json
import hashlib
from collections import deque
from pathlib import Path

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


def _sha256_file(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open('rb') as file:
        for chunk in iter(lambda: file.read(chunk_size), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load_agent_config(checkpoint_dir):
    """Restore the saved configuration for a final GCIQL-Chunk checkpoint."""
    checkpoint_dir = Path(checkpoint_dir)
    saved = json.loads((checkpoint_dir / 'flags.json').read_text())
    saved_agent = saved.get('agent', {})
    name = saved_agent.get('agent_name')
    if name == 'gciql_chunk':
        from agents.gciql_chunk import get_config
    elif name == 'gciql_chunk_lewm':
        from agents.gciql_chunk_lewm import get_config
    else:
        raise ValueError(f'Expected a final GCIQL-Chunk checkpoint, got {name!r}.')

    config = get_config()
    for key, value in saved_agent.items():
        if key in config:
            config[key] = value
    return name, config, saved


def load_lance_policy(lance_path, checkpoint_dir, checkpoint_step):
    """Load a final policy using a LeWM-4Tasks Lance shape probe."""
    from agents import agents
    from agents.gciql_chunk_lewm import LeWMGCIQLChunkAgent
    from utils.datasets import GCChunkDataset
    from utils.flax_utils import restore_agent
    from utils.lewm_dataset import LeWMLanceDataset

    name, config, saved = load_agent_config(checkpoint_dir)
    base = LeWMLanceDataset(lance_path, split='train', validation_fraction=0.05)
    dataset = GCChunkDataset(base, config, preprocess_frame_stack=False)
    example = dataset.sample(1, evaluation=True)

    if name == 'gciql_chunk':
        agent = agents[name].create(
            0,
            example['observations'],
            example['actions'],
            config,
        )
        return restore_agent(agent, checkpoint_dir, checkpoint_step)

    lewm_checkpoint = saved.get('lewm_checkpoint')
    if lewm_checkpoint is None:
        lewm_checkpoint = saved['representation']['lewm_checkpoint']
    from lewm_jax import load_frozen_lewm

    model, variables, metadata = load_frozen_lewm(lewm_checkpoint)
    agent = LeWMGCIQLChunkAgent.create(
        0,
        jnp.asarray(example['observations']),
        jnp.zeros((1, int(metadata['config']['embed_dim'])), dtype=jnp.float32),
        jnp.asarray(example['actions'], dtype=jnp.float32),
        config,
    )
    agent = restore_agent(agent, checkpoint_dir, checkpoint_step)
    encode_pixels = jax.jit(
        lambda pixels: model.apply(
            variables,
            pixels,
            train=False,
            method=model.encode_pixels,
        ).astype(jnp.float32)
    )
    return LeWMEncodedAgent(
        agent,
        encode_pixels,
        share_pi_encoder=config.share_pi_encoder,
        share_q_encoder=config.share_q_encoder,
        lewm_checkpoint=metadata['path'],
    )


class LeWMEncodedAgent:
    """Route actor and Q calls through their configured pixel/LeWM inputs."""

    def __init__(
        self,
        agent,
        encode_pixels,
        share_pi_encoder,
        share_q_encoder=False,
        lewm_checkpoint=None,
    ):
        self.agent = agent
        self.encode_pixels = encode_pixels
        self.share_pi_encoder = bool(share_pi_encoder)
        self.share_q_encoder = bool(share_q_encoder)
        self.lewm_checkpoint = lewm_checkpoint
        self.action_horizon = int(agent.action_horizon)

    def sample_actions(self, observations, goals, seed, temperature):
        if self.share_pi_encoder:
            observations = self.encode_pixels(jnp.asarray(observations))
            goals = self.encode_pixels(jnp.asarray(goals))
        return self.agent.sample_actions(
            observations=observations,
            goals=goals,
            seed=seed,
            temperature=temperature,
        )

    def sample_actions_with_latent_goal(
        self, observations, latent_goals, seed, temperature
    ):
        """Run a shared-LeWM actor with an already encoded goal."""
        if not self.share_pi_encoder:
            raise ValueError(
                'Direct-policy latent subgoals require share_pi_encoder=True.'
            )
        observations = self.encode_pixels(jnp.asarray(observations))
        return self.agent.sample_actions(
            observations=observations,
            goals=jnp.asarray(latent_goals),
            seed=seed,
            temperature=temperature,
        )

class GCIQLChunkPolicy:
    """Execute normalized GCIQL-Chunk actions in LeWM-4Tasks environments."""

    def __init__(self, agent, scaler, seed):
        self.agent = agent
        self.scaler = scaler
        self.rng = jax.random.PRNGKey(seed)
        self.action_horizon = int(agent.action_horizon)
        if self.action_horizon < 1:
            raise ValueError('Policy action_horizon must be positive.')
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
            expected = self.action_horizon * self._action_dim
            if normalized.shape[-1] != expected:
                raise ValueError(
                    f'Policy returned width {normalized.shape[-1]}, expected {expected}.'
                )
            atomic = self.scaler.inverse_transform(
                normalized.reshape(-1, self._action_dim)
            )
            self._chunks = atomic.reshape(
                self._num_envs,
                self.action_horizon,
                self._action_dim,
            )
            self._chunk_index = 0
        actions = self._chunks[:, self._chunk_index].copy()
        self._chunk_index += 1
        actions[~alive] = np.nan
        return actions


class LatentSubgoalGCIQLChunkPolicy:
    """Condition a shared-LeWM direct policy on predicted latent subgoals."""

    def __init__(
        self,
        agent,
        scaler,
        seed,
        latent_subgoal_checkpoint,
        num_samples,
        action_block,
    ):
        if not isinstance(agent, LeWMEncodedAgent) or not agent.share_pi_encoder:
            raise ValueError(
                'Direct-policy latent subgoals require a pi/all checkpoint whose '
                'actor consumes frozen LeWM latents.'
            )
        if int(agent.action_horizon) != int(action_block):
            raise ValueError(
                'Direct policy action horizon must match the configured action block.'
            )
        if num_samples != 1:
            raise ValueError(
                'Latent subgoal inference is temporarily fixed to one sample.'
            )
        (
            subgoal_model,
            subgoal_params,
            subgoal_config,
            checkpoint_step,
        ) = load_latent_subgoal_checkpoint(latent_subgoal_checkpoint)
        expected_sha = subgoal_config.get('lewm_checkpoint_sha256')
        actual_sha = _sha256_file(agent.lewm_checkpoint)
        if expected_sha != actual_sha:
            raise ValueError(
                'Direct policy and latent subgoal generator must use the same '
                'frozen LeWM checkpoint.'
            )
        trained_action_block = int(subgoal_config['action_block'])
        if trained_action_block != int(action_block):
            raise ValueError(
                'Latent subgoal and direct-policy action blocks must match.'
            )

        self.agent = agent
        self.scaler = scaler
        self.seed = int(seed)
        self.lewm_checkpoint = str(Path(agent.lewm_checkpoint).expanduser().resolve())
        self.action_horizon = int(action_block)
        self.latent_subgoal_checkpoint = str(
            Path(latent_subgoal_checkpoint).expanduser().resolve()
        )
        self.latent_subgoal_checkpoint_step = int(checkpoint_step)
        self.latent_subgoal_config = subgoal_config
        self.latent_subgoal_num_samples = int(num_samples)
        self.latent_subgoal_history_size = int(subgoal_config.get('history_size', 1))
        self.latent_subgoal_waypoint_index = None
        self.latent_subgoal_waypoint_step = int(subgoal_config['subgoal_steps'])
        self._latent_subgoal_requires_rng = False

        architecture = subgoal_config['architecture']
        if architecture == FLOW_TRANSFORMER_ARCHITECTURE:
            sampling_steps = int(subgoal_config['flow_sampling_steps'])
            solver = str(subgoal_config['flow_solver'])
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
                        solver=solver,
                    )
                )
            )
        elif architecture == LATENT_PATH_FLOW_ARCHITECTURE:
            sampling_steps = int(subgoal_config['flow_sampling_steps'])
            solver = str(subgoal_config['flow_solver'])
            waypoint_steps = latent_path_waypoint_steps(
                subgoal_config['subgoal_steps'], trained_action_block
            )
            self.latent_subgoal_waypoint_index = len(waypoint_steps) - 1
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
                        solver=solver,
                    )
                )[:, self.latent_subgoal_waypoint_index]
            )
        else:
            self.latent_subgoal_sample_selection = 'deterministic'
            self._predict_latent_subgoal = jax.jit(
                lambda current, goal: subgoal_model.apply(
                    {'params': subgoal_params}, current, goal
                )
            )

    def reset(self, action_space, num_envs):
        self._action_dim = int(np.prod(action_space.shape))
        self._buffers = [deque() for _ in range(num_envs)]
        self._pixel_histories = [
            deque(maxlen=self.latent_subgoal_history_size) for _ in range(num_envs)
        ]
        self.latent_subgoal_generation_counts = np.zeros(num_envs, dtype=np.int64)

    def _predict(self, env_index, goal):
        history = list(self._pixel_histories[env_index])
        history = [history[0]] * (self.latent_subgoal_history_size - len(history)) + history
        history_embeddings = self.agent.encode_pixels(jnp.asarray(np.stack(history)))
        current = (
            history_embeddings[None]
            if self.latent_subgoal_history_size > 1
            else history_embeddings[-1:]
        )
        goal_embedding = self.agent.encode_pixels(jnp.asarray(goal[None]))
        generation = int(self.latent_subgoal_generation_counts[env_index])
        if self._latent_subgoal_requires_rng:
            rng = jax.random.fold_in(jax.random.PRNGKey(self.seed), int(env_index))
            rng = jax.random.fold_in(rng, generation)
            prediction = self._predict_latent_subgoal(current, goal_embedding, rng)
        else:
            prediction = self._predict_latent_subgoal(current, goal_embedding)
        prediction = np.asarray(prediction)[0].astype(np.float32)
        expected_shape = (int(self.latent_subgoal_config['embed_dim']),)
        if prediction.shape != expected_shape or not np.isfinite(prediction).all():
            raise FloatingPointError(
                f'Invalid predicted latent subgoal shape/value: {prediction.shape}.'
            )
        self.latent_subgoal_generation_counts[env_index] += 1
        return prediction

    def get_actions(self, pixels, goals, alive):
        for env_index in np.flatnonzero(alive):
            self._pixel_histories[env_index].append(np.asarray(pixels[env_index, -1]))
            if self._buffers[env_index]:
                continue
            generation = int(self.latent_subgoal_generation_counts[env_index])
            latent_goal = self._predict(env_index, np.asarray(goals[env_index, -1]))
            action_rng = jax.random.fold_in(
                jax.random.PRNGKey(self.seed + 1), int(env_index)
            )
            action_rng = jax.random.fold_in(action_rng, generation)
            normalized = np.asarray(
                self.agent.sample_actions_with_latent_goal(
                    observations=np.asarray(pixels[env_index, -1:]),
                    latent_goals=latent_goal[None],
                    seed=action_rng,
                    temperature=0.0,
                )
            )[0]
            expected_width = self.action_horizon * self._action_dim
            if normalized.shape != (expected_width,):
                raise ValueError(
                    f'Policy returned shape {normalized.shape}, expected '
                    f'({expected_width},).'
                )
            atomic = self.scaler.inverse_transform(
                normalized.reshape(self.action_horizon, self._action_dim)
            )
            self._buffers[env_index].extend(atomic)
        actions = np.full((len(alive), self._action_dim), np.nan, dtype=np.float32)
        for env_index in np.flatnonzero(alive):
            actions[env_index] = self._buffers[env_index].popleft()
        return actions
