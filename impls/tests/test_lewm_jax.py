import jax
import jax.numpy as jnp
import pytest

from lewm_jax import (
    REFERENCE_ARCHITECTURE,
    VARIANT_ARCHITECTURE,
    LeWM,
    architecture_for_encoder,
    build_model,
    lewm_loss,
    loss_for_architecture,
    uses_imagenet_preprocessing,
)
from train_lewm import LeWMConfig


def test_reference_training_defaults():
    config = LeWMConfig()
    assert config.epochs == 10
    assert config.batch_size == 128
    assert config.decode_workers == 6
    assert config.history_size == 3
    assert config.frameskip == 5
    assert config.predictor_depth == 6
    assert config.predictor_heads == 16
    assert config.learning_rate == 5e-5
    assert config.weight_decay == 1e-3
    assert config.sigreg_weight == 0.09
    assert config.architecture == REFERENCE_ARCHITECTURE


def test_encoder_architecture_selection_is_explicit():
    assert architecture_for_encoder('vit_tiny14') == REFERENCE_ARCHITECTURE
    assert architecture_for_encoder('impala_small') == VARIANT_ARCHITECTURE
    assert uses_imagenet_preprocessing({'architecture': REFERENCE_ARCHITECTURE})
    assert not uses_imagenet_preprocessing({'architecture': VARIANT_ARCHITECTURE})


@pytest.mark.parametrize('architecture', [REFERENCE_ARCHITECTURE, VARIANT_ARCHITECTURE])
def test_lewm_encoder_variants_forward_and_loss(architecture):
    image_size = 224 if architecture == REFERENCE_ARCHITECTURE else 64
    encoder = 'vit_tiny14' if architecture == REFERENCE_ARCHITECTURE else 'impala_small'
    model = build_model(
        {'architecture': architecture, 'encoder': encoder, 'image_size': image_size},
        dtype=jnp.float32,
    )
    loss_function = loss_for_architecture(architecture)
    pixel_dtype = jnp.float32 if architecture == REFERENCE_ARCHITECTURE else jnp.uint8
    pixels = jnp.zeros((2, 4, image_size, image_size, 3), dtype=pixel_dtype)
    actions = jnp.zeros((2, 4, 10), dtype=jnp.float32)
    params_key, dropout_key, sigreg_key = jax.random.split(jax.random.PRNGKey(0), 3)
    variables = model.init(
        {'params': params_key, 'dropout': dropout_key}, pixels, actions, train=False
    )

    embeddings, predictions = model.apply(
        variables,
        pixels,
        actions,
        train=False,
        rngs={'dropout': dropout_key},
    )
    loss, (metrics, batch_stats) = loss_function(
        model,
        variables,
        {'pixels': pixels, 'action': actions},
        train=True,
        dropout_key=dropout_key,
        sigreg_key=sigreg_key,
        sigreg_num_proj=8,
    )

    assert embeddings.shape == (2, 4, 192)
    assert predictions.shape == (2, 3, 192)
    assert jnp.isfinite(loss)
    assert all(jnp.isfinite(value) for value in metrics.values())
    assert jax.tree_util.tree_leaves(batch_stats)
