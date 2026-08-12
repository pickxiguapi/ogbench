"""JAX/Flax port of the official LeWM architecture and objective.

The module follows lucas-maes/le-wm at commit ``8edfeb3`` and the
``stable-pretraining==0.1.8`` ViT helper used by that repository.  In
particular, targets are not stop-gradient copies: the encoder is optimized by
both sides of the prediction MSE, exactly as in the reference ``train.py``.
"""

from __future__ import annotations

from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp


def _torch_uniform(fan_in):
    bound = fan_in**-0.5
    return nn.initializers.uniform(scale=bound)


class TorchLinear(nn.Module):
    """Linear layer with PyTorch ``nn.Linear`` default initialization."""

    features: int
    use_bias: bool = True
    dtype: Any = jnp.bfloat16

    @nn.compact
    def __call__(self, x):
        fan_in = x.shape[-1]
        kernel = self.param('kernel', _torch_uniform(fan_in), (fan_in, self.features))
        y = jnp.matmul(x.astype(self.dtype), kernel.astype(self.dtype))
        if self.use_bias:
            bias = self.param('bias', _torch_uniform(fan_in), (self.features,))
            y = y + bias.astype(self.dtype)
        return y


class ViTLinear(nn.Module):
    """HuggingFace ViT linear initialization (normal 0.02, zero bias)."""

    features: int
    use_bias: bool = True
    dtype: Any = jnp.bfloat16

    @nn.compact
    def __call__(self, x):
        kernel = self.param('kernel', nn.initializers.normal(0.02), (x.shape[-1], self.features))
        y = jnp.matmul(x.astype(self.dtype), kernel.astype(self.dtype))
        if self.use_bias:
            bias = self.param('bias', nn.initializers.zeros, (self.features,))
            y = y + bias.astype(self.dtype)
        return y


class ViTSelfAttention(nn.Module):
    dim: int = 192
    heads: int = 3
    dtype: Any = jnp.bfloat16

    @nn.compact
    def __call__(self, x):
        head_dim = self.dim // self.heads
        q = ViTLinear(self.dim, dtype=self.dtype, name='query')(x)
        k = ViTLinear(self.dim, dtype=self.dtype, name='key')(x)
        v = ViTLinear(self.dim, dtype=self.dtype, name='value')(x)
        q = q.reshape(*q.shape[:-1], self.heads, head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(*k.shape[:-1], self.heads, head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(*v.shape[:-1], self.heads, head_dim).transpose(0, 2, 1, 3)
        attention = jnp.einsum('bhqd,bhkd->bhqk', q, k) * (head_dim**-0.5)
        attention = nn.softmax(attention.astype(jnp.float32), axis=-1).astype(self.dtype)
        x = jnp.einsum('bhqk,bhkd->bhqd', attention, v)
        x = x.transpose(0, 2, 1, 3).reshape(x.shape[0], x.shape[2], self.dim)
        return ViTLinear(self.dim, dtype=self.dtype, name='output')(x)


class ViTBlock(nn.Module):
    dim: int = 192
    heads: int = 3
    mlp_dim: int = 768
    dtype: Any = jnp.bfloat16

    @nn.compact
    def __call__(self, x):
        y = nn.LayerNorm(epsilon=1e-12, dtype=self.dtype, name='layernorm_before')(x)
        x = x + ViTSelfAttention(self.dim, self.heads, self.dtype, name='attention')(y)
        y = nn.LayerNorm(epsilon=1e-12, dtype=self.dtype, name='layernorm_after')(x)
        y = ViTLinear(self.mlp_dim, dtype=self.dtype, name='intermediate')(y)
        y = nn.gelu(y, approximate=False)
        y = ViTLinear(self.dim, dtype=self.dtype, name='output')(y)
        return x + y


class ViTTiny14(nn.Module):
    """The exact configuration emitted by stable-pretraining's ``vit_hf``."""

    image_size: int = 224
    patch_size: int = 14
    dim: int = 192
    depth: int = 12
    heads: int = 3
    mlp_dim: int = 768
    dtype: Any = jnp.bfloat16

    @nn.compact
    def __call__(self, pixels):
        if pixels.shape[-3:] != (self.image_size, self.image_size, 3):
            raise ValueError(f'ViT expected (*, {self.image_size}, {self.image_size}, 3), got {pixels.shape}.')
        x = nn.Conv(
            self.dim,
            kernel_size=(self.patch_size, self.patch_size),
            strides=(self.patch_size, self.patch_size),
            padding='VALID',
            kernel_init=nn.initializers.normal(0.02),
            bias_init=nn.initializers.zeros,
            dtype=self.dtype,
            name='patch_embeddings',
        )(pixels)
        x = x.reshape(x.shape[0], -1, self.dim)
        cls_token = self.param('cls_token', nn.initializers.normal(0.02), (1, 1, self.dim))
        cls = jnp.broadcast_to(cls_token.astype(self.dtype), (x.shape[0], 1, self.dim))
        x = jnp.concatenate([cls, x], axis=1)
        position = self.param('position_embeddings', nn.initializers.normal(0.02), (1, x.shape[1], self.dim))
        x = x + position.astype(self.dtype)
        for index in range(self.depth):
            x = ViTBlock(self.dim, self.heads, self.mlp_dim, self.dtype, name=f'layer_{index}')(x)
        return nn.LayerNorm(epsilon=1e-12, dtype=self.dtype, name='layernorm')(x)[:, 0]


class PyTorchBatchNorm1d(nn.Module):
    """BatchNorm1d with PyTorch's unbiased running-variance update."""

    momentum: float = 0.1
    epsilon: float = 1e-5
    dtype: Any = jnp.bfloat16

    @nn.compact
    def __call__(self, x, *, train):
        features = x.shape[-1]
        scale = self.param('scale', nn.initializers.ones, (features,))
        bias = self.param('bias', nn.initializers.zeros, (features,))
        running_mean = self.variable('batch_stats', 'mean', jnp.zeros, (features,), jnp.float32)
        running_var = self.variable('batch_stats', 'var', jnp.ones, (features,), jnp.float32)
        if train:
            x_float = x.astype(jnp.float32)
            axes = tuple(range(x.ndim - 1))
            mean = jnp.mean(x_float, axis=axes)
            variance = jnp.mean((x_float - mean) ** 2, axis=axes)
            sample_count = x.size // features
            unbiased_variance = variance * sample_count / max(sample_count - 1, 1)
            running_mean.value = (1.0 - self.momentum) * running_mean.value + self.momentum * mean
            running_var.value = (1.0 - self.momentum) * running_var.value + self.momentum * unbiased_variance
        else:
            mean, variance = running_mean.value, running_var.value
        normalized = (x.astype(jnp.float32) - mean) * jax.lax.rsqrt(variance + self.epsilon)
        normalized = normalized * scale + bias
        return normalized.astype(self.dtype)


class ProjectionMLP(nn.Module):
    dim: int = 192
    hidden_dim: int = 2048
    dtype: Any = jnp.bfloat16

    @nn.compact
    def __call__(self, x, *, train):
        x = TorchLinear(self.hidden_dim, dtype=self.dtype, name='linear_1')(x)
        x = PyTorchBatchNorm1d(dtype=self.dtype, name='batch_norm')(x, train=train)
        x = nn.gelu(x, approximate=False)
        return TorchLinear(self.dim, dtype=self.dtype, name='linear_2')(x)


class ActionEmbedder(nn.Module):
    dim: int = 192
    smoothed_dim: int = 10
    mlp_scale: int = 4
    dtype: Any = jnp.bfloat16

    @nn.compact
    def __call__(self, actions):
        x = TorchLinear(self.smoothed_dim, dtype=self.dtype, name='patch_embed')(actions.astype(jnp.float32))
        x = TorchLinear(self.mlp_scale * self.dim, dtype=self.dtype, name='linear_1')(x)
        x = nn.silu(x)
        return TorchLinear(self.dim, dtype=self.dtype, name='linear_2')(x)


class PredictorAttention(nn.Module):
    dim: int = 192
    heads: int = 16
    dim_head: int = 64
    dropout: float = 0.1
    dtype: Any = jnp.bfloat16

    @nn.compact
    def __call__(self, x, *, train):
        x = nn.LayerNorm(epsilon=1e-5, dtype=self.dtype, name='norm')(x)
        qkv = TorchLinear(self.heads * self.dim_head * 3, use_bias=False, dtype=self.dtype, name='to_qkv')(x)
        q, k, v = jnp.split(qkv, 3, axis=-1)
        reshape = lambda value: value.reshape(value.shape[0], value.shape[1], self.heads, self.dim_head).transpose(0, 2, 1, 3)
        q, k, v = reshape(q), reshape(k), reshape(v)
        logits = jnp.einsum('bhqd,bhkd->bhqk', q, k) * (self.dim_head**-0.5)
        causal = jnp.tril(jnp.ones((x.shape[1], x.shape[1]), dtype=bool))
        logits = jnp.where(causal[None, None], logits, jnp.finfo(jnp.float32).min)
        attention = nn.softmax(logits.astype(jnp.float32), axis=-1).astype(self.dtype)
        attention = nn.Dropout(rate=self.dropout, name='attention_dropout')(attention, deterministic=not train)
        out = jnp.einsum('bhqk,bhkd->bhqd', attention, v)
        out = out.transpose(0, 2, 1, 3).reshape(x.shape[0], x.shape[1], self.heads * self.dim_head)
        out = TorchLinear(self.dim, dtype=self.dtype, name='to_out')(out)
        return nn.Dropout(rate=self.dropout, name='output_dropout')(out, deterministic=not train)


class PredictorFeedForward(nn.Module):
    dim: int = 192
    hidden_dim: int = 2048
    dropout: float = 0.1
    dtype: Any = jnp.bfloat16

    @nn.compact
    def __call__(self, x, *, train):
        x = nn.LayerNorm(epsilon=1e-5, dtype=self.dtype, name='norm')(x)
        x = TorchLinear(self.hidden_dim, dtype=self.dtype, name='linear_1')(x)
        x = nn.gelu(x, approximate=False)
        x = nn.Dropout(rate=self.dropout, name='dropout_1')(x, deterministic=not train)
        x = TorchLinear(self.dim, dtype=self.dtype, name='linear_2')(x)
        return nn.Dropout(rate=self.dropout, name='dropout_2')(x, deterministic=not train)


class ConditionalBlock(nn.Module):
    dim: int = 192
    heads: int = 16
    dim_head: int = 64
    mlp_dim: int = 2048
    dropout: float = 0.1
    dtype: Any = jnp.bfloat16

    @nn.compact
    def __call__(self, x, condition, *, train):
        modulation = nn.silu(condition)
        modulation = nn.Dense(
            6 * self.dim,
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.zeros,
            dtype=self.dtype,
            name='adaLN_modulation',
        )(modulation)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = jnp.split(modulation, 6, axis=-1)

        y = nn.LayerNorm(
            epsilon=1e-6, use_scale=False, use_bias=False, dtype=self.dtype, name='norm_1'
        )(x)
        y = y * (1 + scale_msa) + shift_msa
        x = x + gate_msa * PredictorAttention(
            self.dim, self.heads, self.dim_head, self.dropout, self.dtype, name='attention'
        )(y, train=train)

        y = nn.LayerNorm(
            epsilon=1e-6, use_scale=False, use_bias=False, dtype=self.dtype, name='norm_2'
        )(x)
        y = y * (1 + scale_mlp) + shift_mlp
        x = x + gate_mlp * PredictorFeedForward(
            self.dim, self.mlp_dim, self.dropout, self.dtype, name='feed_forward'
        )(y, train=train)
        return x


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
        position = self.param('position_embedding', nn.initializers.normal(1.0), (1, self.num_frames, self.dim))
        x = embeddings + position[:, : embeddings.shape[1]].astype(self.dtype)
        x = nn.Dropout(rate=self.emb_dropout, name='embedding_dropout')(x, deterministic=not train)
        for index in range(self.depth):
            x = ConditionalBlock(
                self.dim,
                self.heads,
                self.dim_head,
                self.mlp_dim,
                self.dropout,
                self.dtype,
                name=f'block_{index}',
            )(x, action_embeddings, train=train)
        return nn.LayerNorm(epsilon=1e-5, dtype=self.dtype, name='final_norm')(x)


class LeWM(nn.Module):
    """Reference LeWM encoder, action-conditioned predictor, and projectors."""

    image_size: int = 224
    embed_dim: int = 192
    history_size: int = 3
    dtype: Any = jnp.bfloat16

    def setup(self):
        self.encoder = ViTTiny14(image_size=self.image_size, dim=self.embed_dim, dtype=self.dtype)
        self.projector = ProjectionMLP(self.embed_dim, dtype=self.dtype)
        self.action_encoder = ActionEmbedder(self.embed_dim, dtype=self.dtype)
        self.predictor = ARPredictor(num_frames=self.history_size, dim=self.embed_dim, dtype=self.dtype)
        self.pred_projector = ProjectionMLP(self.embed_dim, dtype=self.dtype)

    def encode_pixels(self, pixels, *, train):
        leading_shape = pixels.shape[:-3]
        flat_pixels = pixels.reshape(-1, *pixels.shape[-3:])
        embeddings = self.encoder(flat_pixels)
        embeddings = self.projector(embeddings, train=train)
        return embeddings.reshape(*leading_shape, self.embed_dim)

    def predict_embeddings(self, embeddings, actions, *, train):
        action_embeddings = self.action_encoder(actions)
        predictions = self.predictor(embeddings, action_embeddings, train=train)
        leading_shape = predictions.shape[:-1]
        predictions = self.pred_projector(predictions.reshape(-1, self.embed_dim), train=train)
        return predictions.reshape(*leading_shape, self.embed_dim)

    def __call__(self, pixels, actions, *, train):
        embeddings = self.encode_pixels(pixels, train=train)
        context_embeddings = embeddings[:, : self.history_size]
        context_actions = actions[:, : self.history_size]
        predictions = self.predict_embeddings(context_embeddings, context_actions, train=train)
        # Lightning runs the complete reference forward/loss under bf16 mixed
        # precision.  In particular, the elementwise prediction MSE keeps the
        # bf16 dtype emitted by the model instead of being promoted to fp32.
        return embeddings, predictions

    def rollout_cost(self, pixels, goals, action_candidates):
        """LeWM's official final-embedding planning cost.

        Args:
            pixels: Normalized context images ``(B, S, H, 224, 224, 3)``.
            goals: Normalized goal images with the same leading ``(B, S)``.
            action_candidates: Normalized blocked actions ``(B, S, T, A)``.
        """
        num_samples = action_candidates.shape[1]
        # CEM expands identical observations across its sample dimension.  The
        # reference implementation encodes sample zero once, then broadcasts.
        context_embeddings = self.encode_pixels(pixels[:, 0], train=False)
        context_embeddings = jnp.broadcast_to(
            context_embeddings[:, None],
            (context_embeddings.shape[0], num_samples, *context_embeddings.shape[1:]),
        )
        goal_embeddings = self.encode_pixels(goals[:, 0], train=False)[..., -1, :]

        history = context_embeddings.shape[2]
        horizon = action_candidates.shape[2]
        if history > horizon:
            raise ValueError(f'Context length {history} exceeds planning horizon {horizon}.')
        actions = action_candidates[:, :, :history]
        future_actions = action_candidates[:, :, history:]
        embeddings = context_embeddings

        batch_size = embeddings.shape[0]
        for step in range(horizon - history):
            flat_embeddings = embeddings.reshape(batch_size * num_samples, embeddings.shape[2], self.embed_dim)
            flat_actions = actions.reshape(batch_size * num_samples, actions.shape[2], actions.shape[3])
            prediction = self.predict_embeddings(
                flat_embeddings[:, -self.history_size :],
                flat_actions[:, -self.history_size :],
                train=False,
            )[:, -1]
            prediction = prediction.reshape(batch_size, num_samples, 1, self.embed_dim)
            embeddings = jnp.concatenate([embeddings, prediction], axis=2)
            actions = jnp.concatenate([actions, future_actions[:, :, step : step + 1]], axis=2)

        flat_embeddings = embeddings.reshape(batch_size * num_samples, embeddings.shape[2], self.embed_dim)
        flat_actions = actions.reshape(batch_size * num_samples, actions.shape[2], actions.shape[3])
        prediction = self.predict_embeddings(
            flat_embeddings[:, -self.history_size :],
            flat_actions[:, -self.history_size :],
            train=False,
        )[:, -1]
        prediction = prediction.reshape(batch_size, num_samples, self.embed_dim)
        return jnp.sum((prediction - goal_embeddings[:, None]) ** 2, axis=-1)


def sigreg_loss(embeddings, key, *, knots=17, num_proj=1024):
    """Official single-device Sketch Isotropic Gaussian Regularizer."""
    projection = embeddings.transpose(1, 0, 2)  # (T, B, D)
    directions = jax.random.normal(key, (projection.shape[-1], num_proj), dtype=jnp.float32)
    directions = directions / jnp.linalg.norm(directions, axis=0, keepdims=True)
    t = jnp.linspace(0.0, 3.0, knots, dtype=jnp.float32)
    dt = 3.0 / (knots - 1)
    edge_indices = jnp.asarray([0, knots - 1], dtype=jnp.int32)
    weights = jnp.full((knots,), 2.0 * dt, dtype=jnp.float32).at[edge_indices].set(dt)
    phi = jnp.exp(-(t**2) / 2.0)
    weights = weights * phi
    # PyTorch creates the random directions and quadrature buffers in fp32,
    # but autocast executes both matrix multiplications in bf16.  The multiply
    # by fp32 ``t`` promotes the trigonometric part back to fp32 in between.
    projected = jnp.einsum(
        'tbd,dp->tbp', projection, directions.astype(projection.dtype)
    )
    x_t = projected[..., None] * t
    error = (jnp.cos(x_t).mean(axis=1) - phi) ** 2 + jnp.sin(x_t).mean(axis=1) ** 2
    statistic = jnp.einsum(
        'tpk,k->tp',
        error.astype(projection.dtype),
        weights.astype(projection.dtype),
    ) * projection.shape[1]
    return statistic.mean()


def lewm_loss(
    model,
    variables,
    batch,
    *,
    train,
    dropout_key,
    sigreg_key,
    sigreg_weight=0.09,
    sigreg_knots=17,
    sigreg_num_proj=1024,
):
    """Return the LeWM scalar loss, metrics, and updated BatchNorm state."""
    apply_kwargs = {'train': train}
    if train:
        (embeddings, predictions), updates = model.apply(
            variables,
            batch['pixels'],
            jnp.nan_to_num(batch['action']),
            rngs={'dropout': dropout_key},
            mutable=['batch_stats'],
            **apply_kwargs,
        )
    else:
        embeddings, predictions = model.apply(
            variables,
            batch['pixels'],
            jnp.nan_to_num(batch['action']),
            rngs={'dropout': dropout_key},
            mutable=False,
            **apply_kwargs,
        )
        updates = {'batch_stats': variables['batch_stats']}
    targets = embeddings[:, 1:]
    prediction_loss = jnp.mean((predictions - targets) ** 2)
    regularizer = sigreg_loss(embeddings, sigreg_key, knots=sigreg_knots, num_proj=sigreg_num_proj)
    loss = prediction_loss + sigreg_weight * regularizer
    metrics = {'loss': loss, 'pred_loss': prediction_loss, 'sigreg_loss': regularizer}
    return loss, (metrics, updates['batch_stats'])
