"""Trainable LeWM predictor modules shared by the IMPALA variant.

Ordinary affine, normalization, dropout, and attention operations use Flax
directly.  The custom modules below only express LeWM-specific compositions:
the projector, action embedder, AdaLN-zero block, and autoregressive predictor.
"""

from __future__ import annotations

from typing import Any

import flax.linen as nn
import jax.numpy as jnp


class ProjectionMLP(nn.Module):
    dim: int = 192
    hidden_dim: int = 2048
    dtype: Any = jnp.bfloat16

    @nn.compact
    def __call__(self, x, *, train):
        fan_in = x.shape[-1]
        x = nn.Dense(
            self.hidden_dim,
            kernel_init=nn.initializers.uniform(scale=fan_in**-0.5),
            bias_init=nn.initializers.uniform(scale=fan_in**-0.5),
            dtype=self.dtype,
            name='linear_1',
        )(x)
        x = nn.BatchNorm(
            use_running_average=not train,
            momentum=0.9,
            epsilon=1e-5,
            dtype=self.dtype,
            name='batch_norm',
        )(x)
        x = nn.gelu(x, approximate=False)
        return nn.Dense(
            self.dim,
            kernel_init=nn.initializers.uniform(scale=self.hidden_dim**-0.5),
            bias_init=nn.initializers.uniform(scale=self.hidden_dim**-0.5),
            dtype=self.dtype,
            name='linear_2',
        )(x)


class ActionEmbedder(nn.Module):
    """Embed one flattened action block at every sequence position."""

    dim: int = 192
    smoothed_dim: int = 10
    mlp_scale: int = 4
    dtype: Any = jnp.bfloat16

    @nn.compact
    def __call__(self, actions):
        # Dense on the last axis is equivalent to the reference kernel-size-1
        # Conv1d without introducing a one-off convolution implementation.
        fan_in = actions.shape[-1]
        x = nn.Dense(
            self.smoothed_dim,
            kernel_init=nn.initializers.uniform(scale=fan_in**-0.5),
            bias_init=nn.initializers.uniform(scale=fan_in**-0.5),
            dtype=self.dtype,
            name='patch_embed',
        )(actions)
        x = nn.Dense(
            self.mlp_scale * self.dim,
            kernel_init=nn.initializers.uniform(scale=self.smoothed_dim**-0.5),
            bias_init=nn.initializers.uniform(scale=self.smoothed_dim**-0.5),
            dtype=self.dtype,
            name='linear_1',
        )(x)
        x = nn.silu(x)
        hidden_dim = self.mlp_scale * self.dim
        return nn.Dense(
            self.dim,
            kernel_init=nn.initializers.uniform(scale=hidden_dim**-0.5),
            bias_init=nn.initializers.uniform(scale=hidden_dim**-0.5),
            dtype=self.dtype,
            name='linear_2',
        )(x)


class ConditionalBlock(nn.Module):
    """Reference predictor block: AdaLN-zero conditioning plus causal attention."""

    dim: int = 192
    heads: int = 16
    dim_head: int = 64
    mlp_dim: int = 2048
    dropout: float = 0.1
    dtype: Any = jnp.bfloat16

    @nn.compact
    def __call__(self, x, condition, *, train):
        modulation = nn.Dense(
            6 * self.dim,
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.zeros,
            dtype=self.dtype,
            name='adaLN_modulation',
        )(nn.silu(condition))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = jnp.split(
            modulation, 6, axis=-1
        )

        y = nn.LayerNorm(
            epsilon=1e-6,
            use_scale=False,
            use_bias=False,
            dtype=self.dtype,
            name='norm_1',
        )(x)
        y = y * (1 + scale_msa) + shift_msa
        # The reference Attention module has its own pre-norm in addition to
        # the block's non-affine AdaLN norm; preserve that computation graph.
        y = nn.LayerNorm(epsilon=1e-5, dtype=self.dtype, name='attention_norm')(y)
        causal_mask = nn.make_causal_mask(jnp.ones(y.shape[:-1], dtype=bool))
        attention = nn.SelfAttention(
            num_heads=self.heads,
            qkv_features=self.heads * self.dim_head,
            out_features=self.dim,
            dropout_rate=self.dropout,
            broadcast_dropout=False,
            use_bias=False,
            kernel_init=nn.initializers.uniform(scale=self.dim**-0.5),
            out_kernel_init=nn.initializers.uniform(
                scale=(self.heads * self.dim_head) ** -0.5
            ),
            force_fp32_for_softmax=True,
            dtype=self.dtype,
            name='attention',
        )(y, mask=causal_mask, deterministic=not train)
        # Flax exposes one bias switch for both QKV and output projections.
        # LeWM uses bias-free QKV and a biased output, so add only that missing
        # output bias while retaining Flax's standard attention operation.
        attention_bias = self.param(
            'attention_output_bias',
            nn.initializers.uniform(scale=(self.heads * self.dim_head) ** -0.5),
            (self.dim,),
        )
        attention = nn.Dropout(rate=self.dropout, name='attention_output_dropout')(
            attention + attention_bias.astype(self.dtype), deterministic=not train
        )
        x = x + gate_msa * attention

        y = nn.LayerNorm(
            epsilon=1e-6,
            use_scale=False,
            use_bias=False,
            dtype=self.dtype,
            name='norm_2',
        )(x)
        y = y * (1 + scale_mlp) + shift_mlp
        # As in the reference FeedForward, this is a second, affine pre-norm.
        y = nn.LayerNorm(epsilon=1e-5, dtype=self.dtype, name='feed_forward_norm')(y)
        y = nn.Dense(
            self.mlp_dim,
            kernel_init=nn.initializers.uniform(scale=self.dim**-0.5),
            bias_init=nn.initializers.uniform(scale=self.dim**-0.5),
            dtype=self.dtype,
            name='feed_forward_in',
        )(y)
        y = nn.gelu(y, approximate=False)
        y = nn.Dropout(rate=self.dropout, name='feed_forward_dropout_in')(
            y, deterministic=not train
        )
        y = nn.Dense(
            self.dim,
            kernel_init=nn.initializers.uniform(scale=self.mlp_dim**-0.5),
            bias_init=nn.initializers.uniform(scale=self.mlp_dim**-0.5),
            dtype=self.dtype,
            name='feed_forward_out',
        )(y)
        y = nn.Dropout(rate=self.dropout, name='feed_forward_dropout_out')(
            y, deterministic=not train
        )
        return x + gate_mlp * y


class ARPredictor(nn.Module):
    num_frames: int = 3
    dim: int = 192
    depth: int = 6
    heads: int = 16
    dim_head: int = 64
    mlp_dim: int = 2048
    dropout: float = 0.1
    emb_dropout: float = 0.0
    dtype: Any = jnp.bfloat16

    @nn.compact
    def __call__(self, embeddings, action_embeddings, *, train):
        position = self.param(
            'position_embedding', nn.initializers.normal(1.0), (1, self.num_frames, self.dim)
        )
        x = embeddings + position[:, : embeddings.shape[1]].astype(self.dtype)
        x = nn.Dropout(rate=self.emb_dropout, name='embedding_dropout')(
            x, deterministic=not train
        )
        for index in range(self.depth):
            x = ConditionalBlock(
                dim=self.dim,
                heads=self.heads,
                dim_head=self.dim_head,
                mlp_dim=self.mlp_dim,
                dropout=self.dropout,
                dtype=self.dtype,
                name=f'block_{index}',
            )(x, action_embeddings, train=train)
        return nn.LayerNorm(epsilon=1e-5, dtype=self.dtype, name='final_norm')(x)
