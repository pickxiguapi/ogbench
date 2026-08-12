"""Single construction path for trainable LeWM JAX models."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from typing import Any

import jax.numpy as jnp

from lewm_jax.impala import ImpalaLeWM, lewm_loss as impala_lewm_loss
from lewm_jax.vit import LeWM as ViTLeWM
from lewm_jax.vit import lewm_loss as vit_lewm_loss

REFERENCE_ARCHITECTURE = 'reference_vit_66d47b6'
VARIANT_ARCHITECTURE = 'lewm_impala_variant'


def _config_dict(config) -> dict[str, Any]:
    if is_dataclass(config):
        return asdict(config)
    if isinstance(config, Mapping):
        return dict(config)
    raise TypeError(f'Expected a dataclass or mapping config, got {type(config)!r}.')


def architecture_for_encoder(encoder: str) -> str:
    if encoder == 'vit_tiny14':
        return REFERENCE_ARCHITECTURE
    if encoder == 'impala_small':
        return VARIANT_ARCHITECTURE
    raise ValueError(f'Unknown LeWM encoder: {encoder}')


def build_model(config, *, dtype=jnp.bfloat16):
    """Build exactly the model described by a train/checkpoint config."""
    values = _config_dict(config)
    architecture = values.get('architecture')
    if architecture is None:
        architecture = architecture_for_encoder(values.get('encoder', 'vit_tiny14'))
    common = dict(
        image_size=int(values.get('image_size', 224)),
        embed_dim=int(values.get('embed_dim', 192)),
        history_size=int(values.get('history_size', 3)),
        dtype=dtype,
    )
    if architecture == REFERENCE_ARCHITECTURE:
        return ViTLeWM(**common)
    if architecture == VARIANT_ARCHITECTURE:
        return ImpalaLeWM(
            **common,
            projector_hidden_dim=int(values.get('projector_hidden_dim', 2048)),
            action_smoothed_dim=int(values.get('action_smoothed_dim', 10)),
            action_mlp_scale=int(values.get('action_mlp_scale', 4)),
            predictor_depth=int(values.get('predictor_depth', 6)),
            predictor_heads=int(values.get('predictor_heads', 16)),
            predictor_dim_head=int(values.get('predictor_dim_head', 64)),
            predictor_mlp_dim=int(values.get('predictor_mlp_dim', 2048)),
            predictor_dropout=float(values.get('predictor_dropout', 0.1)),
            predictor_emb_dropout=float(values.get('predictor_emb_dropout', 0.0)),
        )
    raise ValueError(f'Unsupported LeWM checkpoint architecture: {architecture}')


def loss_for_architecture(architecture: str):
    if architecture == REFERENCE_ARCHITECTURE:
        return vit_lewm_loss
    if architecture == VARIANT_ARCHITECTURE:
        return impala_lewm_loss
    raise ValueError(f'Unsupported LeWM loss architecture: {architecture}')


def uses_imagenet_preprocessing(config) -> bool:
    values = _config_dict(config)
    architecture = values.get('architecture')
    if architecture is None:
        architecture = architecture_for_encoder(values.get('encoder', 'vit_tiny14'))
    return architecture == REFERENCE_ARCHITECTURE
