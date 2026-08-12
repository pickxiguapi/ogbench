import jax
import jax.numpy as jnp
import flax.linen as nn
import pytest

from lewm_jax import LeWM, lewm_loss
from lewm_jax.encoders import memory_efficient_dot_product_attention
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


def test_fused_attention_cpu_matches_reference_attention():
    query = jax.random.normal(jax.random.PRNGKey(0), (2, 7, 3, 8))
    key = jax.random.normal(jax.random.PRNGKey(1), (2, 7, 3, 8))
    value = jax.random.normal(jax.random.PRNGKey(2), (2, 7, 3, 8))
    expected = nn.dot_product_attention(query, key, value, deterministic=True)
    actual = memory_efficient_dot_product_attention(
        query, key, value, deterministic=True
    )
    assert jnp.allclose(actual, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize(
    ('encoder', 'image_size'),
    [
        ('impala_small', 64),
        ('vit_tiny14', 224),
    ],
)
def test_lewm_encoder_variants_forward_and_loss(encoder, image_size):
    model = LeWM(encoder_name=encoder, dtype=jnp.float32)
    pixels = jnp.zeros((2, 4, image_size, image_size, 3), dtype=jnp.uint8)
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
    loss, (metrics, batch_stats) = lewm_loss(
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
