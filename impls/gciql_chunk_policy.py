"""Checkpoint loading and execution adapters for final GCIQL-Chunk policies."""

from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np


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

    def score_actions(self, observations, goals, actions):
        if self.share_q_encoder:
            observations = self.encode_pixels(jnp.asarray(observations))
            goals = self.encode_pixels(jnp.asarray(goals))
        return self.agent.score_actions(observations, goals, actions)


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
