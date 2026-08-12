"""Trainable Flax LeWM components for reference ViT and OGBench encoders.

The JEPA/predictor/rollout organization follows ``dhidary/le-wm-jax`` at
commit e52c1a0, whose inference port is parity-tested against released LeWM
checkpoints. ``lewm_jax.reference`` preserves the Server-23-proven reference
ViT training backend. The default exports below are the OGBench variants.
"""

from lewm_jax.losses import lewm_loss, sigreg_loss
from lewm_jax.model import LeWM

__all__ = ['LeWM', 'lewm_loss', 'sigreg_loss']
