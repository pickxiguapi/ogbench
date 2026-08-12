"""Visual encoders supported by the trainable LeWM implementation."""

from __future__ import annotations

from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp

from utils.encoders import encoder_modules

IMAGENET_MEAN = jnp.asarray([0.485, 0.456, 0.406], dtype=jnp.float32)
IMAGENET_STD = jnp.asarray([0.229, 0.224, 0.225], dtype=jnp.float32)


def memory_efficient_dot_product_attention(
    query,
    key,
    value,
    bias=None,
    mask=None,
    *,
    broadcast_dropout=True,
    dropout_rng=None,
    dropout_rate=0.0,
    deterministic=False,
    dtype=None,
    precision=None,
    module=None,
    force_fp32_for_softmax=False,
    **unused_kwargs,
):
    """Use fused SDPA for ViT without materializing the full attention map.

    LeWM's ViT attention has no dropout, bias, or mask.  The extra keyword
    arguments are accepted because this function is passed through Flax's
    ``SelfAttention`` interface.  On CUDA, JAX dispatches explicitly to the
    cuDNN fused forward/backward implementation; CPU tests use the XLA path.
    """
    del (
        broadcast_dropout,
        dropout_rng,
        dtype,
        precision,
        module,
        force_fp32_for_softmax,
        unused_kwargs,
    )
    if dropout_rate and not deterministic:
        raise ValueError('Fused LeWM ViT attention does not support attention dropout.')
    if jax.default_backend() != 'gpu':
        return jax.nn.dot_product_attention(
            query, key, value, bias=bias, mask=mask, implementation='xla'
        )

    if bias is not None or mask is not None:
        raise ValueError('LeWM ViT fused attention expects no bias or mask.')

    # ViT-Tiny/14 has 256 patch tokens plus CLS = 257. cuDNN fused attention
    # requires an aligned physical sequence length on RTX 3090. Pad the
    # storage to a multiple of 8 while passing the true lengths, so padded
    # keys never participate and the returned first 257 rows are exactly the
    # original attention problem.
    query_len = query.shape[1]
    key_len = key.shape[1]
    padded_query_len = ((query_len + 7) // 8) * 8
    padded_key_len = ((key_len + 7) // 8) * 8
    query = jnp.pad(query, ((0, 0), (0, padded_query_len - query_len), (0, 0), (0, 0)))
    key = jnp.pad(key, ((0, 0), (0, padded_key_len - key_len), (0, 0), (0, 0)))
    value = jnp.pad(value, ((0, 0), (0, padded_key_len - key_len), (0, 0), (0, 0)))
    query_lengths = jnp.full((query.shape[0],), query_len, dtype=jnp.int32)
    key_lengths = jnp.full((key.shape[0],), key_len, dtype=jnp.int32)
    output = jax.nn.dot_product_attention(
        query,
        key,
        value,
        query_seq_lengths=query_lengths,
        key_value_seq_lengths=key_lengths,
        implementation='cudnn',
    )
    return output[:, :query_len]


class ViTBlock(nn.Module):
    """HuggingFace-style pre-norm ViT block built from Flax primitives."""

    dim: int = 192
    heads: int = 3
    mlp_dim: int = 768
    dtype: Any = jnp.bfloat16

    @nn.compact
    def __call__(self, x):
        normal_init = nn.initializers.normal(0.02)
        y = nn.LayerNorm(epsilon=1e-12, dtype=self.dtype, name='layernorm_before')(x)
        y = nn.SelfAttention(
            num_heads=self.heads,
            qkv_features=self.dim,
            out_features=self.dim,
            kernel_init=normal_init,
            bias_init=nn.initializers.zeros,
            force_fp32_for_softmax=True,
            attention_fn=memory_efficient_dot_product_attention,
            dtype=self.dtype,
            name='attention',
        )(y)
        x = x + y
        y = nn.LayerNorm(epsilon=1e-12, dtype=self.dtype, name='layernorm_after')(x)
        y = nn.Dense(
            self.mlp_dim,
            kernel_init=normal_init,
            bias_init=nn.initializers.zeros,
            dtype=self.dtype,
            name='intermediate',
        )(y)
        y = nn.gelu(y, approximate=False)
        y = nn.Dense(
            self.dim,
            kernel_init=normal_init,
            bias_init=nn.initializers.zeros,
            dtype=self.dtype,
            name='output',
        )(y)
        return x + y


class ViTTiny14(nn.Module):
    """ViT-Tiny/14 used by the original LeWM encoder."""

    image_size: int = 224
    patch_size: int = 14
    dim: int = 192
    depth: int = 12
    heads: int = 3
    mlp_dim: int = 768
    dtype: Any = jnp.bfloat16

    @nn.compact
    def __call__(self, pixels, *, train=False):
        del train
        if pixels.shape[-3:] != (self.image_size, self.image_size, 3):
            raise ValueError(
                f'ViT expected (*, {self.image_size}, {self.image_size}, 3), got {pixels.shape}.'
            )
        # The public model interface is uniformly raw uint8 RGB/NHWC. Keep the
        # encoder-specific transform inside the encoder so training, planning,
        # and evaluation cannot accidentally disagree about preprocessing.
        pixels = pixels.astype(jnp.float32) / 255.0
        pixels = (pixels - IMAGENET_MEAN) / IMAGENET_STD
        normal_init = nn.initializers.normal(0.02)
        x = nn.Conv(
            self.dim,
            kernel_size=(self.patch_size, self.patch_size),
            strides=(self.patch_size, self.patch_size),
            padding='VALID',
            kernel_init=normal_init,
            bias_init=nn.initializers.zeros,
            dtype=self.dtype,
            name='patch_embeddings',
        )(pixels)
        x = x.reshape(x.shape[0], -1, self.dim)
        cls_token = self.param('cls_token', normal_init, (1, 1, self.dim))
        cls = jnp.broadcast_to(cls_token.astype(self.dtype), (x.shape[0], 1, self.dim))
        x = jnp.concatenate([cls, x], axis=1)
        position = self.param(
            'position_embeddings', normal_init, (1, x.shape[1], self.dim)
        )
        x = x + position.astype(self.dtype)
        for index in range(self.depth):
            x = ViTBlock(
                dim=self.dim,
                heads=self.heads,
                mlp_dim=self.mlp_dim,
                dtype=self.dtype,
                name=f'layer_{index}',
            )(x)
        return nn.LayerNorm(epsilon=1e-12, dtype=self.dtype, name='layernorm')(x)[:, 0]


def make_encoder(name, *, image_size, embed_dim, patch_size, dtype):
    if name == 'vit_tiny14':
        return ViTTiny14(
            image_size=image_size,
            patch_size=patch_size,
            dim=embed_dim,
            dtype=dtype,
        )
    if name == 'impala_small':
        return encoder_modules['impala_small']()
    raise ValueError(f'Unknown LeWM encoder: {name}')
