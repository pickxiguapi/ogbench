"""Shared inference runtime for frozen-latent subgoal generators."""

from __future__ import annotations

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
)


class LatentSubgoalGenerator:
    """Load one subgoal checkpoint and generate replanning targets."""

    def __init__(
        self,
        checkpoint,
        encode_pixels,
        *,
        seed,
        action_block,
        num_samples=1,
        lewm_checkpoint=None,
    ):
        if int(num_samples) != 1:
            raise ValueError('Latent subgoal inference is temporarily fixed to one sample.')

        model, params, config, checkpoint_step = load_latent_subgoal_checkpoint(
            checkpoint
        )
        expected_sha = config.get('lewm_checkpoint_sha256')
        if expected_sha is not None:
            if lewm_checkpoint is None:
                raise ValueError(
                    'Latent subgoal inference requires its frozen LeWM checkpoint.'
                )
            digest = hashlib.sha256()
            with Path(lewm_checkpoint).expanduser().open('rb') as file:
                for chunk in iter(lambda: file.read(8 * 1024 * 1024), b''):
                    digest.update(chunk)
            if digest.hexdigest() != expected_sha:
                raise ValueError(
                    'Latent subgoal generator and controller/policy must use the '
                    'same frozen LeWM checkpoint.'
                )
        trained_action_block = int(config['action_block'])
        if trained_action_block != int(action_block):
            raise ValueError(
                'Latent subgoal and controller action blocks must match: '
                f'{trained_action_block} != {action_block}.'
            )

        self.checkpoint = str(Path(checkpoint).expanduser().resolve())
        self.checkpoint_step = int(checkpoint_step)
        self.config = config
        self.encode_pixels = encode_pixels
        self.seed = int(seed)
        self.num_samples = int(num_samples)
        self.embed_dim = int(config['embed_dim'])
        self.waypoint_step = int(config['subgoal_steps'])
        self.waypoint_index = None
        self.history_size = 1
        self.sample_selection = 'deterministic'
        self._requires_rng = False

        architecture = config['architecture']
        if architecture == FLOW_TRANSFORMER_ARCHITECTURE:
            sampling_steps = int(config['flow_sampling_steps'])
            solver = str(config['flow_solver'])
            self.sample_selection = 'single_sample'
            self._requires_rng = True
            self._predict = jax.jit(
                lambda current, goal, rng: sample_conditional_flow_candidates(
                    model,
                    params,
                    current,
                    goal,
                    rng,
                    num_samples=1,
                    num_steps=sampling_steps,
                    solver=solver,
                )[:, 0]
            )
        elif architecture == LATENT_PATH_FLOW_ARCHITECTURE:
            sampling_steps = int(config['flow_sampling_steps'])
            solver = str(config['flow_solver'])
            self.history_size = int(config.get('history_size', 1))
            if self.history_size <= 0:
                raise ValueError('Latent subgoal history size must be positive.')
            waypoint_steps = latent_path_waypoint_steps(
                self.waypoint_step, trained_action_block
            )
            self.waypoint_index = waypoint_steps.index(self.waypoint_step)
            self.sample_selection = 'single_sample'
            self._requires_rng = True
            self._predict = jax.jit(
                lambda current, goal, rng: sample_conditional_path_flow_candidates(
                    model,
                    params,
                    current,
                    goal,
                    rng,
                    num_samples=1,
                    num_steps=sampling_steps,
                    solver=solver,
                )[:, 0, self.waypoint_index]
            )
        else:
            self._predict = jax.jit(
                lambda current, goal: model.apply(
                    {'params': params}, current, goal
                )
            )

        self.histories = []
        self.generation_counts = np.zeros(0, dtype=np.int64)

    def reset(self, num_envs):
        self.histories = [deque(maxlen=self.history_size) for _ in range(num_envs)]
        self.generation_counts = np.zeros(num_envs, dtype=np.int64)

    def observe(self, env_index, pixels):
        self.histories[env_index].append(np.asarray(pixels))

    def predict(self, env_index, goal_pixels):
        history = list(self.histories[env_index])
        if not history:
            raise ValueError('A current observation is required before subgoal prediction.')
        history = [history[0]] * (self.history_size - len(history)) + history
        history_embeddings = np.asarray(
            self.encode_pixels(jnp.asarray(np.stack(history)))
        )
        current = (
            history_embeddings[None]
            if self.history_size > 1
            else history_embeddings[-1:]
        )
        goal = np.asarray(
            self.encode_pixels(jnp.asarray(np.asarray(goal_pixels)[None]))
        )

        generation = int(self.generation_counts[env_index])
        if self._requires_rng:
            rng = jax.random.fold_in(jax.random.PRNGKey(self.seed), int(env_index))
            rng = jax.random.fold_in(rng, generation)
            prediction = self._predict(jnp.asarray(current), jnp.asarray(goal), rng)
        else:
            prediction = self._predict(jnp.asarray(current), jnp.asarray(goal))

        prediction = np.asarray(prediction)[0].astype(np.float32)
        if prediction.shape != (self.embed_dim,) or not np.isfinite(prediction).all():
            raise FloatingPointError(
                f'Invalid predicted latent subgoal shape/value: {prediction.shape}.'
            )
        self.generation_counts[env_index] += 1
        return prediction
