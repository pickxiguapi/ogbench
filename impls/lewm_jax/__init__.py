"""Trainable Flax LeWM components for OGBench visual encoders.

The JEPA/predictor/rollout organization follows ``dhidary/le-wm-jax`` at
commit e52c1a0, whose inference port is parity-tested against released LeWM
checkpoints.  This package replaces the original ViT with OGBench's shared
``impala_small`` encoder and uses trainable Flax primitives throughout, so it
is an OGBench LeWM variant rather than a checkpoint-compatible port.
"""

from lewm_jax.losses import lewm_loss, sigreg_loss
from lewm_jax.model import LeWM

__all__ = ['LeWM', 'lewm_loss', 'sigreg_loss']
