"""Load the final-goal-conditioned GCIQL-Chunk action prior for LeWM++."""

from __future__ import annotations

import hashlib
import json
from collections import deque
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np


def load_action_prior(lance_path, checkpoint_dir, checkpoint_step, lewm_checkpoint=None):
    """Restore the paper's shared-LeWM action prior and frozen encoder."""
    from agents.gciql_chunk_lewm import LeWMGCIQLChunkAgent, get_config
    from lewm_jax import load_frozen_lewm
    from utils.datasets import GCChunkDataset
    from utils.flax_utils import restore_agent
    from utils.lewm_dataset import LeWMLanceDataset

    checkpoint_dir = Path(checkpoint_dir).expanduser().resolve()
    flags_path = checkpoint_dir / 'flags.json'
    if not flags_path.is_file():
        raise FileNotFoundError(f'Action-prior metadata not found: {flags_path}')
    saved = json.loads(flags_path.read_text())
    saved_agent = saved.get('agent', {})
    representation = saved.get('representation', {})
    if saved_agent.get('agent_name') != 'gciql_chunk_lewm':
        raise ValueError('LeWM++ requires a GCIQL-Chunk-LeWM checkpoint.')
    if representation.get('mode') != 'all':
        raise ValueError('LeWM++ requires the shared-all action-prior representation.')

    config = get_config()
    for key, value in saved_agent.items():
        if key in config:
            config[key] = value
    if not (config.share_q_encoder and config.share_v_encoder and config.share_pi_encoder):
        raise ValueError('Action-prior checkpoint does not share all LeWM encoders.')

    base = LeWMLanceDataset(lance_path, split='train', validation_fraction=0.05)
    dataset = GCChunkDataset(base, config, preprocess_frame_stack=False)
    example = dataset.sample(1, evaluation=True)
    lewm_checkpoint = lewm_checkpoint or saved.get('lewm_checkpoint') or representation.get('lewm_checkpoint')
    if lewm_checkpoint is None:
        raise ValueError('A frozen LeWM checkpoint is required to restore the action prior.')
    expected_sha = saved.get('lewm_checkpoint_sha256')
    if expected_sha is not None:
        actual_sha = hashlib.sha256(Path(lewm_checkpoint).expanduser().read_bytes()).hexdigest()
        if actual_sha != expected_sha:
            raise ValueError('Action prior and evaluator must use the same frozen LeWM checkpoint.')
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
    return FinalGoalActionPrior(agent, encode_pixels, metadata['path'])


class FinalGoalActionPrior:
    """Adapter exposing deterministic normalized action chunks to CEM."""

    def __init__(self, agent, encode_pixels, lewm_checkpoint):
        self.agent = agent
        self.encode_pixels = encode_pixels
        self.lewm_checkpoint = lewm_checkpoint
        self.action_horizon = int(agent.action_horizon)

    def sample_actions(self, observations, goals, seed, temperature=0.0):
        observations = self.encode_pixels(jnp.asarray(observations))
        goals = self.encode_pixels(jnp.asarray(goals))
        return self.agent.sample_actions(
            observations=observations,
            goals=goals,
            seed=seed,
            temperature=temperature,
        )


class FinalGoalPolicy:
    """Execute GCIQL-Chunk-AWR directly, always conditioned on the final goal."""

    def __init__(self, action_prior, scaler, seed):
        self.action_prior = action_prior
        self.scaler = scaler
        self.seed = int(seed)
        self.action_horizon = int(action_prior.action_horizon)

    def reset(self, action_space, num_envs):
        self.action_dim = int(np.prod(action_space.shape))
        self.buffers = [deque() for _ in range(num_envs)]
        self.chunk_counts = np.zeros(num_envs, dtype=np.int64)

    def get_actions(self, pixels, goals, alive):
        for env_index in np.flatnonzero(alive):
            if self.buffers[env_index]:
                continue
            key = jax.random.fold_in(jax.random.PRNGKey(self.seed), int(env_index))
            key = jax.random.fold_in(key, int(self.chunk_counts[env_index]))
            normalized = np.asarray(
                self.action_prior.sample_actions(
                    observations=np.asarray(pixels[env_index, -1:]),
                    goals=np.asarray(goals[env_index, -1:]),
                    seed=key,
                    temperature=0.0,
                )
            )[0]
            expected = self.action_horizon * self.action_dim
            if normalized.shape != (expected,):
                raise ValueError(f'Action prior returned {normalized.shape}; expected ({expected},).')
            actions = self.scaler.inverse_transform(normalized.reshape(self.action_horizon, self.action_dim))
            self.buffers[env_index].extend(actions)
            self.chunk_counts[env_index] += 1

        result = np.full((len(alive), self.action_dim), np.nan, dtype=np.float32)
        for env_index in np.flatnonzero(alive):
            result[env_index] = self.buffers[env_index].popleft()
        return result
