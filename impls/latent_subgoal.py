"""Frozen-LeWM latent subgoal model and inference checkpoint loader."""

from __future__ import annotations

import json
from pathlib import Path

import flax
import flax.linen as nn
import jax.numpy as jnp


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
    if config.get('architecture') != 'direct_latent_mlp_512x3':
        raise ValueError(
            f'Unsupported latent subgoal architecture: {config.get("architecture")!r}.'
        )
    if config.get('loss') != 'raw_latent_mse':
        raise ValueError(f'Unsupported latent subgoal loss: {config.get("loss")!r}.')
    embed_dim = int(config['embed_dim'])
    hidden_dims = tuple(int(value) for value in config['hidden_dims'])
    model = LatentSubgoalMLP(embed_dim=embed_dim, hidden_dims=hidden_dims)

    payload = flax.serialization.msgpack_restore(path.read_bytes())
    if set(payload) != {'rng', 'step', 'train_state'}:
        raise ValueError(f'Unexpected latent subgoal checkpoint keys: {set(payload)}.')
    state = payload['train_state']
    if 'params' not in state or int(state['step']) != int(payload['step']):
        raise ValueError(f'Invalid latent subgoal train state in {path}.')
    if int(payload['step']) <= 0:
        raise ValueError(f'Latent subgoal checkpoint step must be positive: {path}.')
    return model, state['params'], config, int(payload['step'])
