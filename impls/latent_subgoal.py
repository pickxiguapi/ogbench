"""Frozen-LeWM latent subgoal models and inference checkpoint loader."""

from __future__ import annotations

import json
from pathlib import Path

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp


DIRECT_MLP_ARCHITECTURE = 'direct_latent_mlp_512x3'
FLOW_TRANSFORMER_ARCHITECTURE = 'latent_flow_transformer_encoder'
LATENT_PATH_FLOW_ARCHITECTURE = 'latent_path_flow_transformer_encoder'


def latent_path_waypoint_steps(subgoal_steps, action_block):
    """Derive path prediction offsets from the control chunk granularity."""
    subgoal_steps = int(subgoal_steps)
    action_block = int(action_block)
    if subgoal_steps <= 0 or action_block <= 0:
        raise ValueError('Subgoal steps and action block must be positive.')
    if subgoal_steps % action_block:
        raise ValueError('Subgoal steps must be divisible by the action block.')
    return tuple(range(action_block, subgoal_steps + 1, action_block))


class LatentSubgoalMLP(nn.Module):
    embed_dim: int
    hidden_dims: tuple[int, ...] = (512, 512, 512)

    @nn.compact
    def __call__(self, current_latents, goal_latents):
        x = jnp.concatenate((current_latents, goal_latents), axis=-1)
        for hidden_dim in self.hidden_dims:
            x = nn.Dense(hidden_dim)(x)
            x = nn.LayerNorm()(x)
            x = nn.silu(x)
        return nn.Dense(self.embed_dim)(x)


def sinusoidal_time_embedding(times, dim):
    """Return transformer-width sinusoidal embeddings for scalar flow times."""
    if dim % 2:
        raise ValueError('The flow-time embedding dimension must be even.')
    times = jnp.asarray(times, dtype=jnp.float32).reshape(-1, 1)
    frequencies = jnp.exp(
        -jnp.log(10_000.0)
        * jnp.arange(dim // 2, dtype=jnp.float32)
        / max(dim // 2 - 1, 1)
    )
    angles = times * frequencies[None]
    return jnp.concatenate((jnp.sin(angles), jnp.cos(angles)), axis=-1)


class TransformerEncoderBlock(nn.Module):
    model_dim: int
    num_heads: int
    mlp_dim: int

    @nn.compact
    def __call__(self, tokens):
        residual = tokens
        tokens = nn.LayerNorm(name='attention_norm')(tokens)
        tokens = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.model_dim,
            out_features=self.model_dim,
            dropout_rate=0.0,
            deterministic=True,
            name='self_attention',
        )(tokens)
        tokens = residual + tokens

        residual = tokens
        tokens = nn.LayerNorm(name='mlp_norm')(tokens)
        tokens = nn.Dense(self.mlp_dim, name='mlp_in')(tokens)
        tokens = nn.gelu(tokens, approximate=False)
        tokens = nn.Dense(self.model_dim, name='mlp_out')(tokens)
        return residual + tokens


class AdaLNTransformerEncoderBlock(nn.Module):
    """Pre-norm Transformer block modulated by one global condition vector."""

    model_dim: int
    num_heads: int
    mlp_dim: int

    @nn.compact
    def __call__(self, tokens, condition):
        modulation = nn.Dense(
            4 * self.model_dim,
            kernel_init=nn.initializers.normal(stddev=0.02),
            bias_init=nn.initializers.zeros_init(),
            name='condition_modulation',
        )(nn.silu(condition))
        attention_scale, attention_shift, mlp_scale, mlp_shift = jnp.split(
            modulation, 4, axis=-1
        )

        residual = tokens
        tokens = nn.LayerNorm(
            use_scale=False, use_bias=False, name='attention_norm'
        )(tokens)
        tokens = tokens * (1.0 + attention_scale[:, None]) + attention_shift[:, None]
        tokens = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.model_dim,
            out_features=self.model_dim,
            dropout_rate=0.0,
            deterministic=True,
            name='self_attention',
        )(tokens)
        tokens = residual + tokens

        residual = tokens
        tokens = nn.LayerNorm(use_scale=False, use_bias=False, name='mlp_norm')(
            tokens
        )
        tokens = tokens * (1.0 + mlp_scale[:, None]) + mlp_shift[:, None]
        tokens = nn.Dense(self.mlp_dim, name='mlp_in')(tokens)
        tokens = nn.gelu(tokens, approximate=False)
        tokens = nn.Dense(self.model_dim, name='mlp_out')(tokens)
        return residual + tokens


class LatentSubgoalFlowTransformer(nn.Module):
    """Conditional vector field over frozen LeWM subgoal latents."""

    embed_dim: int
    model_dim: int = 384
    num_layers: int = 8
    num_heads: int = 8
    mlp_dim: int = 1536

    @nn.compact
    def __call__(self, noisy_latents, current_latents, goal_latents, flow_times):
        noisy_token = nn.Dense(self.model_dim, name='noisy_projection')(noisy_latents)
        current_token = nn.Dense(self.model_dim, name='current_projection')(
            current_latents
        )
        goal_token = nn.Dense(self.model_dim, name='goal_projection')(goal_latents)
        time_features = sinusoidal_time_embedding(flow_times, self.model_dim)
        time_token = nn.Dense(self.model_dim, name='time_projection')(time_features)
        tokens = jnp.stack(
            (current_token, goal_token, noisy_token, time_token), axis=1
        )
        token_types = self.param(
            'token_types', nn.initializers.normal(stddev=0.02), (4, self.model_dim)
        )
        tokens = tokens + token_types[None]
        for layer_index in range(self.num_layers):
            tokens = TransformerEncoderBlock(
                model_dim=self.model_dim,
                num_heads=self.num_heads,
                mlp_dim=self.mlp_dim,
                name=f'encoder_block_{layer_index}',
            )(tokens)
        noisy_token = nn.LayerNorm(name='output_norm')(tokens[:, 2])
        return nn.Dense(self.embed_dim, name='velocity_head')(noisy_token)


class LatentPathFlow(nn.Module):
    """LeFlow-style conditional vector field over a short latent waypoint path."""

    embed_dim: int
    num_waypoints: int = 2
    hidden_dim: int = 512
    depth: int = 4
    num_heads: int = 8
    ff_dim: int = 2048
    time_dim: int = 64
    history_size: int = 1

    @nn.compact
    def __call__(self, noisy_path, current_latents, goal_latents, flow_times):
        if noisy_path.ndim != 3:
            raise ValueError('LatentPathFlow expects noisy_path with shape [B, T, D].')
        if noisy_path.shape[1] != self.num_waypoints:
            raise ValueError(
                f'Expected {self.num_waypoints} waypoints, got {noisy_path.shape[1]}.'
            )

        if self.history_size <= 0:
            raise ValueError('LatentPathFlow history_size must be positive.')
        if self.history_size > 1:
            if current_latents.ndim != 3:
                raise ValueError(
                    'History-conditioned LatentPathFlow expects current_latents '
                    'with shape [B, H, D].'
                )
            if current_latents.shape[1] != self.history_size:
                raise ValueError(
                    f'Expected {self.history_size} history frames, got '
                    f'{current_latents.shape[1]}.'
                )

            path_tokens = nn.Dense(self.hidden_dim, name='path_projection')(
                noisy_path
            )
            path_positions = self.param(
                'path_position_embeddings',
                nn.initializers.normal(stddev=0.02),
                (self.num_waypoints, self.hidden_dim),
            )
            path_tokens = path_tokens + path_positions[None]

            # Keep history, goal, and flow time distinct until a nonlinear fusion
            # layer builds the global condition used by every AdaLN block.
            history_condition = nn.Dense(
                self.hidden_dim, name='history_condition_projection'
            )(current_latents.reshape(current_latents.shape[0], -1))
            goal_condition = nn.Dense(
                self.hidden_dim, name='goal_condition_projection'
            )(goal_latents)
            time_features = sinusoidal_time_embedding(flow_times, self.time_dim)
            time_condition = nn.Dense(
                self.hidden_dim, name='time_projection_in'
            )(time_features)
            time_condition = nn.silu(time_condition)
            time_condition = nn.Dense(
                self.hidden_dim, name='time_projection_out'
            )(time_condition)
            condition = jnp.concatenate(
                (history_condition, goal_condition, time_condition), axis=-1
            )
            condition = nn.Dense(
                self.hidden_dim, name='condition_fusion_in'
            )(condition)
            condition = nn.silu(condition)
            condition = nn.Dense(
                self.hidden_dim, name='condition_fusion_out'
            )(condition)

            for layer_index in range(self.depth):
                path_tokens = AdaLNTransformerEncoderBlock(
                    model_dim=self.hidden_dim,
                    num_heads=self.num_heads,
                    mlp_dim=self.ff_dim,
                    name=f'encoder_block_{layer_index}',
                )(path_tokens, condition)
            path_tokens = nn.LayerNorm(name='output_norm')(path_tokens)
            return nn.Dense(self.embed_dim, name='velocity_head')(path_tokens)

        # Preserve the original single-frame parameter tree so existing checkpoints
        # remain exactly loadable when their config has no history_size field.
        tokens = nn.Dense(self.hidden_dim, name='token_projection')(noisy_path)
        positions = self.param(
            'position_embeddings',
            nn.initializers.normal(stddev=0.02),
            (self.num_waypoints, self.hidden_dim),
        )
        tokens = tokens + positions[None]

        start_condition = nn.Dense(self.hidden_dim, name='start_projection')(
            current_latents
        )
        goal_condition = nn.Dense(self.hidden_dim, name='goal_projection')(
            goal_latents
        )
        time_features = sinusoidal_time_embedding(flow_times, self.time_dim)
        time_condition = nn.Dense(self.hidden_dim, name='time_projection_in')(
            time_features
        )
        time_condition = nn.silu(time_condition)
        time_condition = nn.Dense(self.hidden_dim, name='time_projection_out')(
            time_condition
        )
        tokens = tokens + (
            start_condition + goal_condition + time_condition
        )[:, None]

        for layer_index in range(self.depth):
            tokens = TransformerEncoderBlock(
                model_dim=self.hidden_dim,
                num_heads=self.num_heads,
                mlp_dim=self.ff_dim,
                name=f'encoder_block_{layer_index}',
            )(tokens)
        tokens = nn.LayerNorm(name='output_norm')(tokens)
        return nn.Dense(self.embed_dim, name='velocity_head')(tokens)


def sample_conditional_flow(
    model,
    params,
    current_latents,
    goal_latents,
    rng,
    *,
    num_steps=16,
    solver='heun',
):
    """Integrate a learned conditional flow from N(0, I) to subgoal latents."""
    if num_steps <= 0:
        raise ValueError('Flow sampling steps must be positive.')
    if solver not in ('euler', 'heun'):
        raise ValueError(f'Unsupported flow solver: {solver!r}.')
    current_latents = jnp.asarray(current_latents, dtype=jnp.float32)
    goal_latents = jnp.asarray(goal_latents, dtype=jnp.float32)
    samples = jax.random.normal(rng, current_latents.shape, dtype=jnp.float32)
    step_size = jnp.asarray(1.0 / num_steps, dtype=jnp.float32)

    def integrate_step(index, value):
        flow_time = jnp.full(
            (value.shape[0],), index.astype(jnp.float32) * step_size
        )
        velocity = model.apply(
            {'params': params}, value, current_latents, goal_latents, flow_time
        )
        proposal = value + step_size * velocity
        if solver == 'euler':
            return proposal
        next_time = jnp.minimum(flow_time + step_size, 1.0)
        next_velocity = model.apply(
            {'params': params},
            proposal,
            current_latents,
            goal_latents,
            next_time,
        )
        return value + 0.5 * step_size * (velocity + next_velocity)

    return jax.lax.fori_loop(0, num_steps, integrate_step, samples)


def sample_conditional_flow_candidates(
    model,
    params,
    current_latents,
    goal_latents,
    rng,
    *,
    num_samples,
    num_steps=16,
    solver='heun',
):
    """Draw multiple conditional single-subgoal latents per conditioning pair."""
    if num_samples <= 0:
        raise ValueError('Latent subgoal sample count must be positive.')
    current_latents = jnp.asarray(current_latents, dtype=jnp.float32)
    goal_latents = jnp.asarray(goal_latents, dtype=jnp.float32)
    batch_size = current_latents.shape[0]
    repeated_current = jnp.repeat(current_latents, num_samples, axis=0)
    repeated_goal = jnp.repeat(goal_latents, num_samples, axis=0)
    samples = sample_conditional_flow(
        model,
        params,
        repeated_current,
        repeated_goal,
        rng,
        num_steps=num_steps,
        solver=solver,
    )
    return samples.reshape(batch_size, num_samples, current_latents.shape[-1])


def sample_conditional_path_flow(
    model,
    params,
    current_latents,
    goal_latents,
    rng,
    *,
    num_steps=16,
    solver='euler',
):
    """Integrate a learned path flow from Gaussian noise to latent waypoints."""
    if num_steps <= 0:
        raise ValueError('Flow sampling steps must be positive.')
    if solver not in ('euler', 'heun'):
        raise ValueError(f'Unsupported flow solver: {solver!r}.')
    current_latents = jnp.asarray(current_latents, dtype=jnp.float32)
    goal_latents = jnp.asarray(goal_latents, dtype=jnp.float32)
    sample_shape = (
        current_latents.shape[0],
        int(model.num_waypoints),
        current_latents.shape[-1],
    )
    samples = jax.random.normal(rng, sample_shape, dtype=jnp.float32)
    step_size = jnp.asarray(1.0 / num_steps, dtype=jnp.float32)

    def integrate_step(index, value):
        flow_time = jnp.full(
            (value.shape[0],), index.astype(jnp.float32) * step_size
        )
        velocity = model.apply(
            {'params': params}, value, current_latents, goal_latents, flow_time
        )
        proposal = value + step_size * velocity
        if solver == 'euler':
            return proposal
        next_time = jnp.minimum(flow_time + step_size, 1.0)
        next_velocity = model.apply(
            {'params': params},
            proposal,
            current_latents,
            goal_latents,
            next_time,
        )
        return value + 0.5 * step_size * (velocity + next_velocity)

    return jax.lax.fori_loop(0, num_steps, integrate_step, samples)


def sample_conditional_path_flow_candidates(
    model,
    params,
    current_latents,
    goal_latents,
    rng,
    *,
    num_samples,
    num_steps=16,
    solver='euler',
):
    """Draw multiple conditional latent paths for every conditioning pair."""
    if num_samples <= 0:
        raise ValueError('Latent subgoal sample count must be positive.')
    current_latents = jnp.asarray(current_latents, dtype=jnp.float32)
    goal_latents = jnp.asarray(goal_latents, dtype=jnp.float32)
    batch_size = current_latents.shape[0]
    repeated_current = jnp.repeat(current_latents, num_samples, axis=0)
    repeated_goal = jnp.repeat(goal_latents, num_samples, axis=0)
    paths = sample_conditional_path_flow(
        model,
        params,
        repeated_current,
        repeated_goal,
        rng,
        num_steps=num_steps,
        solver=solver,
    )
    return paths.reshape(
        batch_size,
        num_samples,
        int(model.num_waypoints),
        current_latents.shape[-1],
    )


def select_latent_path_medoid(candidate_paths):
    """Select the sampled path with minimum mean distance to all samples."""
    candidate_paths = jnp.asarray(candidate_paths, dtype=jnp.float32)
    if candidate_paths.ndim != 4:
        raise ValueError(
            'Candidate paths must have shape [B, num_samples, waypoints, D].'
        )
    flat_paths = candidate_paths.reshape(
        candidate_paths.shape[0], candidate_paths.shape[1], -1
    )
    pairwise_squared_distances = jnp.sum(
        jnp.square(flat_paths[:, :, None] - flat_paths[:, None, :]), axis=-1
    )
    medoid_indices = jnp.argmin(
        jnp.mean(pairwise_squared_distances, axis=-1), axis=-1
    )
    return jnp.take_along_axis(
        candidate_paths,
        medoid_indices[:, None, None, None],
        axis=1,
    )[:, 0]


def select_latent_medoid(candidate_latents):
    """Select the sampled latent with minimum mean distance to all samples."""
    candidate_latents = jnp.asarray(candidate_latents, dtype=jnp.float32)
    if candidate_latents.ndim != 3:
        raise ValueError('Candidate latents must have shape [B, num_samples, D].')
    pairwise_squared_distances = jnp.sum(
        jnp.square(
            candidate_latents[:, :, None] - candidate_latents[:, None, :]
        ),
        axis=-1,
    )
    medoid_indices = jnp.argmin(
        jnp.mean(pairwise_squared_distances, axis=-1), axis=-1
    )
    return jnp.take_along_axis(
        candidate_latents, medoid_indices[:, None, None], axis=1
    )[:, 0]


def load_latent_subgoal_checkpoint(path):
    """Load generator parameters and its adjacent immutable run config."""
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f'Latent subgoal checkpoint not found: {path}')
    config_path = path.parent / 'config.json'
    if not config_path.is_file():
        raise FileNotFoundError(
            f'Latent subgoal config must be adjacent to its checkpoint: {config_path}'
        )
    config = json.loads(config_path.read_text())
    architecture = config.get('architecture')
    embed_dim = int(config['embed_dim'])
    if architecture == DIRECT_MLP_ARCHITECTURE:
        if config.get('loss') != 'raw_latent_mse':
            raise ValueError(
                f'Unsupported latent subgoal loss: {config.get("loss")!r}.'
            )
        hidden_dims = tuple(int(value) for value in config['hidden_dims'])
        model = LatentSubgoalMLP(embed_dim=embed_dim, hidden_dims=hidden_dims)
    elif architecture == FLOW_TRANSFORMER_ARCHITECTURE:
        if config.get('loss') != 'conditional_flow_matching_mse':
            raise ValueError(
                f'Unsupported latent subgoal loss: {config.get("loss")!r}.'
            )
        model = LatentSubgoalFlowTransformer(
            embed_dim=embed_dim,
            model_dim=int(config['model_dim']),
            num_layers=int(config['num_layers']),
            num_heads=int(config['num_heads']),
            mlp_dim=int(config['mlp_dim']),
        )
    elif architecture == LATENT_PATH_FLOW_ARCHITECTURE:
        if config.get('loss') != 'conditional_path_flow_matching_mse':
            raise ValueError(
                f'Unsupported latent path flow loss: {config.get("loss")!r}.'
            )
        history_size = int(config.get('history_size', 1))
        if (
            history_size > 1
            and config.get('conditioning') != 'history_goal_time_adaln'
        ):
            raise ValueError(
                'History-conditioned latent path flow requires '
                'conditioning="history_goal_time_adaln".'
            )
        waypoint_steps = latent_path_waypoint_steps(
            config['subgoal_steps'], config['action_block']
        )
        model = LatentPathFlow(
            embed_dim=embed_dim,
            num_waypoints=len(waypoint_steps),
            hidden_dim=int(config['hidden_dim']),
            depth=int(config['depth']),
            num_heads=int(config['num_heads']),
            ff_dim=int(config['ff_dim']),
            time_dim=int(config['time_dim']),
            history_size=history_size,
        )
    else:
        raise ValueError(f'Unsupported latent subgoal architecture: {architecture!r}.')

    payload = flax.serialization.msgpack_restore(path.read_bytes())
    if set(payload) != {'rng', 'step', 'train_state'}:
        raise ValueError(f'Unexpected latent subgoal checkpoint keys: {set(payload)}.')
    state = payload['train_state']
    if 'params' not in state or int(state['step']) != int(payload['step']):
        raise ValueError(f'Invalid latent subgoal train state in {path}.')
    if int(payload['step']) <= 0:
        raise ValueError(f'Latent subgoal checkpoint step must be positive: {path}.')
    params = state.get('ema_params', state['params'])
    return model, params, config, int(payload['step'])
