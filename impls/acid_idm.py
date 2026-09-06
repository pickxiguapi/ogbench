"""ACID-style inverse-dynamics verifier for frozen LeWM latents."""

from __future__ import annotations

from pathlib import Path

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp

from latent_subgoal import sinusoidal_time_embedding


ARCHITECTURE = 'acid_prefix_suffix_flow_idm'


class PrefixSuffixBlock(nn.Module):
    """Transformer block whose latent prefix cannot attend to the action suffix."""

    model_dim: int
    num_heads: int
    mlp_dim: int

    @nn.compact
    def __call__(self, tokens):
        # Rows are queries and columns are keys.  The two latent prefix tokens
        # attend only to the prefix; the noisy-action suffix attends to all tokens.
        mask = jnp.asarray(
            [
                [True, True, False],
                [True, True, False],
                [True, True, True],
            ],
            dtype=jnp.bool_,
        )[None, None]

        residual = tokens
        tokens = nn.LayerNorm(name='attention_norm')(tokens)
        tokens = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.model_dim,
            out_features=self.model_dim,
            dropout_rate=0.0,
            deterministic=True,
            name='self_attention',
        )(tokens, mask=mask)
        tokens = residual + tokens

        residual = tokens
        tokens = nn.LayerNorm(name='mlp_norm')(tokens)
        tokens = nn.Dense(self.mlp_dim, name='mlp_in')(tokens)
        tokens = nn.gelu(tokens, approximate=False)
        tokens = nn.Dense(self.model_dim, name='mlp_out')(tokens)
        return residual + tokens


class ACIDInverseDynamicsFlow(nn.Module):
    """Flow-matching IDM following the ACID appendix architecture."""

    embed_dim: int
    action_dim: int
    model_dim: int = 192
    num_layers: int = 4
    num_heads: int = 3
    mlp_dim: int = 768

    @nn.compact
    def __call__(self, noisy_actions, current_latents, next_latents, flow_times):
        current_token = nn.Dense(self.model_dim, name='current_projection')(
            current_latents
        )
        next_token = nn.Dense(self.model_dim, name='next_projection')(next_latents)
        action_token = nn.Dense(self.model_dim, name='action_projection')(
            noisy_actions
        )
        time_features = sinusoidal_time_embedding(flow_times, self.model_dim)
        action_token = action_token + nn.Dense(
            self.model_dim, name='time_projection'
        )(time_features)
        tokens = jnp.stack((current_token, next_token, action_token), axis=1)
        token_types = self.param(
            'token_types', nn.initializers.normal(stddev=0.02), (3, self.model_dim)
        )
        tokens = tokens + token_types[None]
        for layer_index in range(self.num_layers):
            tokens = PrefixSuffixBlock(
                model_dim=self.model_dim,
                num_heads=self.num_heads,
                mlp_dim=self.mlp_dim,
                name=f'block_{layer_index}',
            )(tokens)
        suffix = nn.LayerNorm(name='output_norm')(tokens[:, 2])
        return nn.Dense(self.action_dim, name='velocity_head')(suffix)


def sample_inverse_actions(model, params, current_latents, next_latents, rng, *, num_steps=1):
    """Integrate ACID's flow from Gaussian noise at tau=1 to actions at tau=0."""
    if num_steps <= 0:
        raise ValueError('num_steps must be positive.')
    current_latents = jnp.asarray(current_latents, dtype=jnp.float32)
    next_latents = jnp.asarray(next_latents, dtype=jnp.float32)
    if current_latents.shape != next_latents.shape or current_latents.ndim != 2:
        raise ValueError('Current and next latents must be same-shape rank-two arrays.')
    values = jax.random.normal(
        rng, (len(current_latents), int(model.action_dim)), dtype=jnp.float32
    )
    step_size = jnp.asarray(1.0 / num_steps, dtype=jnp.float32)

    def integrate_step(index, value):
        tau = jnp.full(
            (value.shape[0],),
            1.0 - index.astype(jnp.float32) * step_size,
            dtype=jnp.float32,
        )
        velocity = model.apply(
            {'params': params}, value, current_latents, next_latents, tau
        )
        return value - step_size * velocity

    return jax.lax.fori_loop(0, num_steps, integrate_step, values)


def load_acid_idm_checkpoint(path):
    """Restore an ACID IDM checkpoint and its action normalization statistics."""
    path = Path(path).expanduser().resolve()
    payload = flax.serialization.msgpack_restore(path.read_bytes())
    config = payload['config']
    if config.get('architecture') != ARCHITECTURE:
        raise ValueError(
            f'Checkpoint architecture {config.get("architecture")!r} is not '
            f'{ARCHITECTURE!r}.'
        )
    model = ACIDInverseDynamicsFlow(
        embed_dim=int(config['embed_dim']),
        action_dim=int(config['action_chunk_dim']),
        model_dim=int(config['model_dim']),
        num_layers=int(config['num_layers']),
        num_heads=int(config['num_heads']),
        mlp_dim=int(config['mlp_dim']),
    )
    return (
        model,
        payload['params'],
        jnp.asarray(payload['action_mean'], dtype=jnp.float32),
        jnp.asarray(payload['action_std'], dtype=jnp.float32),
        config,
        int(payload['step']),
    )
