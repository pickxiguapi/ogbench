"""The three checkpoint-compatible subgoal generators used by LeWM++."""

from __future__ import annotations

import json
from pathlib import Path

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp

MLP_ARCHITECTURE = 'history_latent_mlp'
DIRECT_MLP_ARCHITECTURE = 'direct_latent_mlp_512x3'
ENDPOINT_FLOW_ARCHITECTURE = 'latent_endpoint_flow_transformer_encoder'
LATENT_PATH_FLOW_ARCHITECTURE = 'latent_path_flow_transformer_encoder'
FLOW_CONDITIONING = 'history_goal_time_adaln'

GENERATOR_ARCHITECTURES = {
    'mlp': (MLP_ARCHITECTURE, DIRECT_MLP_ARCHITECTURE),
    'endpoint_flow': (ENDPOINT_FLOW_ARCHITECTURE,),
    'latent_path_flow': (LATENT_PATH_FLOW_ARCHITECTURE,),
}


class LatentSubgoalMLP(nn.Module):
    """Predict one endpoint latent from observation history and final goal."""

    embed_dim: int
    hidden_dims: tuple[int, ...] = (512, 512, 512)

    @nn.compact
    def __call__(self, history_latents, goal_latents):
        history_latents = history_latents.reshape(history_latents.shape[0], -1)
        features = jnp.concatenate((history_latents, goal_latents), axis=-1)
        for hidden_dim in self.hidden_dims:
            features = nn.Dense(hidden_dim)(features)
            features = nn.LayerNorm()(features)
            features = nn.silu(features)
        return nn.Dense(self.embed_dim)(features)


def waypoint_steps(subgoal_steps, action_block):
    """Return the chunk-aligned waypoint offsets predicted by the generator."""
    subgoal_steps = int(subgoal_steps)
    action_block = int(action_block)
    if subgoal_steps <= 0 or action_block <= 0:
        raise ValueError('Subgoal steps and action block must be positive.')
    if subgoal_steps % action_block:
        raise ValueError('Subgoal steps must be divisible by the action block.')
    return tuple(range(action_block, subgoal_steps + 1, action_block))


def sinusoidal_time_embedding(times, dim):
    """Embed unit-interval flow times with fixed sinusoidal features."""
    if dim <= 0 or dim % 2:
        raise ValueError('The flow-time embedding dimension must be positive and even.')
    times = jnp.asarray(times, dtype=jnp.float32).reshape(-1, 1)
    frequencies = jnp.exp(-jnp.log(10_000.0) * jnp.arange(dim // 2, dtype=jnp.float32) / max(dim // 2 - 1, 1))
    angles = times * frequencies[None]
    return jnp.concatenate((jnp.sin(angles), jnp.cos(angles)), axis=-1)


class AdaLNTransformerBlock(nn.Module):
    """Pre-norm Transformer block modulated by one global condition."""

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
        attn_scale, attn_shift, mlp_scale, mlp_shift = jnp.split(modulation, 4, axis=-1)

        residual = tokens
        tokens = nn.LayerNorm(use_scale=False, use_bias=False, name='attention_norm')(tokens)
        tokens = tokens * (1.0 + attn_scale[:, None]) + attn_shift[:, None]
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
        tokens = nn.LayerNorm(use_scale=False, use_bias=False, name='mlp_norm')(tokens)
        tokens = tokens * (1.0 + mlp_scale[:, None]) + mlp_shift[:, None]
        tokens = nn.Dense(self.mlp_dim, name='mlp_in')(tokens)
        tokens = nn.gelu(tokens, approximate=False)
        tokens = nn.Dense(self.model_dim, name='mlp_out')(tokens)
        return residual + tokens


class LatentPathFlow(nn.Module):
    """Conditional vector field over a chunk-aligned latent waypoint path."""

    embed_dim: int
    num_waypoints: int = 2
    hidden_dim: int = 512
    depth: int = 4
    num_heads: int = 8
    ff_dim: int = 2048
    time_dim: int = 64
    history_size: int = 3

    @nn.compact
    def __call__(self, noisy_path, history_latents, goal_latents, flow_times):
        if noisy_path.ndim != 3 or noisy_path.shape[1] != self.num_waypoints:
            raise ValueError(f'noisy_path must have shape [B, {self.num_waypoints}, {self.embed_dim}].')
        if history_latents.ndim != 3 or history_latents.shape[1] != self.history_size:
            raise ValueError(f'history_latents must have shape [B, {self.history_size}, {self.embed_dim}].')

        path_tokens = nn.Dense(self.hidden_dim, name='path_projection')(noisy_path)
        positions = self.param(
            'path_position_embeddings',
            nn.initializers.normal(stddev=0.02),
            (self.num_waypoints, self.hidden_dim),
        )
        path_tokens = path_tokens + positions[None]

        history_condition = nn.Dense(self.hidden_dim, name='history_condition_projection')(
            history_latents.reshape(history_latents.shape[0], -1)
        )
        goal_condition = nn.Dense(self.hidden_dim, name='goal_condition_projection')(goal_latents)
        time_condition = nn.Dense(self.hidden_dim, name='time_projection_in')(
            sinusoidal_time_embedding(flow_times, self.time_dim)
        )
        time_condition = nn.Dense(self.hidden_dim, name='time_projection_out')(nn.silu(time_condition))
        condition = nn.Dense(self.hidden_dim, name='condition_fusion_in')(
            jnp.concatenate((history_condition, goal_condition, time_condition), axis=-1)
        )
        condition = nn.Dense(self.hidden_dim, name='condition_fusion_out')(nn.silu(condition))

        for layer_index in range(self.depth):
            path_tokens = AdaLNTransformerBlock(
                model_dim=self.hidden_dim,
                num_heads=self.num_heads,
                mlp_dim=self.ff_dim,
                name=f'encoder_block_{layer_index}',
            )(path_tokens, condition)
        path_tokens = nn.LayerNorm(name='output_norm')(path_tokens)
        return nn.Dense(self.embed_dim, name='velocity_head')(path_tokens)


def sample_path(model, params, history_latents, goal_latents, rng, *, num_steps=16):
    """Integrate the learned flow from Gaussian noise with Euler steps."""
    if num_steps <= 0:
        raise ValueError('Flow sampling steps must be positive.')
    history_latents = jnp.asarray(history_latents, dtype=jnp.float32)
    goal_latents = jnp.asarray(goal_latents, dtype=jnp.float32)
    sample_shape = (
        history_latents.shape[0],
        int(model.num_waypoints),
        history_latents.shape[-1],
    )
    samples = jax.random.normal(rng, sample_shape, dtype=jnp.float32)
    step_size = jnp.asarray(1.0 / num_steps, dtype=jnp.float32)

    def integrate_step(index, value):
        flow_time = jnp.full((value.shape[0],), index.astype(jnp.float32) * step_size)
        velocity = model.apply({'params': params}, value, history_latents, goal_latents, flow_time)
        return value + step_size * velocity

    return jax.lax.fori_loop(0, num_steps, integrate_step, samples)


def sample_path_candidates(
    model,
    params,
    history_latents,
    goal_latents,
    rng,
    *,
    num_samples,
    num_steps=16,
):
    """Draw multiple complete paths for every conditioning pair."""
    if num_samples <= 0:
        raise ValueError('Latent path sample count must be positive.')
    history_latents = jnp.asarray(history_latents, dtype=jnp.float32)
    goal_latents = jnp.asarray(goal_latents, dtype=jnp.float32)
    batch_size = history_latents.shape[0]
    paths = sample_path(
        model,
        params,
        jnp.repeat(history_latents, num_samples, axis=0),
        jnp.repeat(goal_latents, num_samples, axis=0),
        rng,
        num_steps=num_steps,
    )
    return paths.reshape(
        batch_size,
        num_samples,
        int(model.num_waypoints),
        history_latents.shape[-1],
    )


def select_path_medoid(candidate_paths):
    """Select the sampled path nearest to all samples in squared distance."""
    candidate_paths = jnp.asarray(candidate_paths, dtype=jnp.float32)
    if candidate_paths.ndim != 4:
        raise ValueError('Candidate paths must have shape [B, num_samples, waypoints, D].')
    flat_paths = candidate_paths.reshape(candidate_paths.shape[0], candidate_paths.shape[1], -1)
    distances = jnp.sum(jnp.square(flat_paths[:, :, None] - flat_paths[:, None, :]), axis=-1)
    medoid_indices = jnp.argmin(jnp.mean(distances, axis=-1), axis=-1)
    return jnp.take_along_axis(candidate_paths, medoid_indices[:, None, None, None], axis=1)[:, 0]


def load_checkpoint(path):
    """Load an MLP, Endpoint Flow, or LatentPath Flow checkpoint."""
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f'Subgoal-generator checkpoint not found: {path}')
    config_path = path.parent / 'config.json'
    if not config_path.is_file():
        raise FileNotFoundError(f'Subgoal-generator config must be adjacent to its checkpoint: {config_path}')
    config = json.loads(config_path.read_text())
    architecture = config.get('architecture')
    history_size = int(config.get('history_size', 1))
    if history_size <= 0:
        raise ValueError('Generator history size must be positive.')
    if architecture in (MLP_ARCHITECTURE, DIRECT_MLP_ARCHITECTURE):
        if config.get('loss') != 'raw_latent_mse':
            raise ValueError('MLP checkpoints must use raw_latent_mse.')
        if architecture == MLP_ARCHITECTURE and config.get('conditioning') != 'history_goal':
            raise ValueError('History MLP checkpoints must use history_goal conditioning.')
        model = LatentSubgoalMLP(
            embed_dim=int(config['embed_dim']),
            hidden_dims=tuple(int(value) for value in config['hidden_dims']),
        )
    elif architecture in (ENDPOINT_FLOW_ARCHITECTURE, LATENT_PATH_FLOW_ARCHITECTURE):
        expected_loss = (
            'conditional_endpoint_flow_matching_mse'
            if architecture == ENDPOINT_FLOW_ARCHITECTURE
            else 'conditional_path_flow_matching_mse'
        )
        if config.get('loss') != expected_loss or config.get('conditioning') != FLOW_CONDITIONING:
            raise ValueError('Flow checkpoint loss/conditioning does not match its architecture.')
        if config.get('flow_solver') != 'euler':
            raise ValueError('LeWM++ flow checkpoints must use Euler integration.')
        steps = (
            (int(config['subgoal_steps']),)
            if architecture == ENDPOINT_FLOW_ARCHITECTURE
            else waypoint_steps(config['subgoal_steps'], config['action_block'])
        )
        model = LatentPathFlow(
            embed_dim=int(config['embed_dim']),
            num_waypoints=len(steps),
            hidden_dim=int(config['hidden_dim']),
            depth=int(config['depth']),
            num_heads=int(config['num_heads']),
            ff_dim=int(config['ff_dim']),
            time_dim=int(config['time_dim']),
            history_size=history_size,
        )
    else:
        raise ValueError(f'Unsupported generator architecture: {architecture!r}.')

    payload = flax.serialization.msgpack_restore(path.read_bytes())
    if set(payload) != {'rng', 'step', 'train_state'}:
        raise ValueError(f'Unexpected subgoal-generator checkpoint keys: {set(payload)}.')
    state = payload['train_state']
    if 'params' not in state or int(state['step']) != int(payload['step']):
        raise ValueError(f'Invalid subgoal-generator train state in {path}.')
    if int(payload['step']) <= 0:
        raise ValueError(f'Subgoal-generator checkpoint step must be positive: {path}.')
    return model, state.get('ema_params', state['params']), config, int(payload['step'])
