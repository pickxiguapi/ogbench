"""Final trainable LeWM JAX API.

The default exports are the Server-23-proven reference ViT implementation.
The OGBench IMPALA implementation remains available under explicit ``Variant``
names so it cannot be selected accidentally for a reproduction run.
"""

from lewm_jax.factory import (
    REFERENCE_ARCHITECTURE,
    VARIANT_ARCHITECTURE,
    architecture_for_encoder,
    build_model,
    loss_for_architecture,
    uses_imagenet_preprocessing,
)
from lewm_jax.losses import lewm_loss as variant_lewm_loss
from lewm_jax.model import LeWM as VariantLeWM
from lewm_jax.reference import LeWM, lewm_loss, sigreg_loss

__all__ = [
    'LeWM',
    'lewm_loss',
    'sigreg_loss',
    'VariantLeWM',
    'variant_lewm_loss',
    'REFERENCE_ARCHITECTURE',
    'VARIANT_ARCHITECTURE',
    'architecture_for_encoder',
    'build_model',
    'loss_for_architecture',
    'uses_imagenet_preprocessing',
]
