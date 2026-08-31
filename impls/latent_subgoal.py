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
    angles = times * frequencies[None] * 1000.0
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
