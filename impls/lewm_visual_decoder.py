"""LeWorldModel-compatible latent-to-RGB decoder for visualization only."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn


class CrossAttention(nn.Module):
    def __init__(self, dim=256, heads=8, dim_head=64, dropout=0.1):
        super().__init__()
        inner_dim = heads * dim_head
        self.heads = heads
        self.dropout = dropout
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, 2 * inner_dim, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, query, memory):
        query = self.norm_q(query)
        memory = self.norm_kv(memory)
        q = rearrange(self.to_q(query), 'b p (h d) -> b h p d', h=self.heads)
        k, v = self.to_kv(memory).chunk(2, dim=-1)
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.heads)
        dropout = self.dropout if self.training else 0.0
        output = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout)
        return self.to_out(rearrange(output, 'b h p d -> b p (h d)'))


class DecoderBlock(nn.Module):
    def __init__(self, dim=256, heads=8, dim_head=64, mlp_dim=512, dropout=0.1):
        super().__init__()
        self.cross_attention = CrossAttention(dim, heads, dim_head, dropout)
        self.mlp = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, query, memory):
        query = query + self.cross_attention(query, memory)
        return query + self.mlp(query)


class CLSDecoder(nn.Module):
    """Official LeWM-style CLS decoder: ``[B,192] -> [B,3,224,224]``."""

    def __init__(
        self,
        cls_dim=192,
        hidden_dim=256,
        depth=4,
        heads=8,
        dim_head=64,
        mlp_dim=512,
        dropout=0.1,
        image_size=224,
        patch_size=16,
    ):
        super().__init__()
        if image_size % patch_size:
            raise ValueError('image_size must be divisible by patch_size.')
        self.image_size = image_size
        self.patch_size = patch_size
        num_patches = (image_size // patch_size) ** 2
        self.cls_projection = nn.Linear(cls_dim, hidden_dim)
        self.query_tokens = nn.Parameter(torch.empty(1, num_patches, hidden_dim))
        self.blocks = nn.ModuleList(
            [DecoderBlock(hidden_dim, heads, dim_head, mlp_dim, dropout) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.patch_head = nn.Linear(hidden_dim, patch_size * patch_size * 3)
        nn.init.trunc_normal_(self.query_tokens, std=0.02)

    def forward(self, cls_embedding):
        memory = self.cls_projection(cls_embedding).unsqueeze(1)
        query = self.query_tokens.expand(cls_embedding.shape[0], -1, -1)
        for block in self.blocks:
            query = block(query, memory)
        patches = self.patch_head(self.norm(query))
        side = self.image_size // self.patch_size
        image = rearrange(
            patches,
            'b (h w) (p1 p2 c) -> b c (h p1) (w p2)',
            h=side,
            w=side,
            p1=self.patch_size,
            p2=self.patch_size,
            c=3,
        )
        return torch.tanh(image)


class LegacyConvDecoder(nn.Module):
    """Compatibility loader for the first local convolutional probe."""

    def __init__(
        self,
        cls_dim=192,
        base_dim=512,
        init_size=7,
        image_size=224,
        ch_mult=(1, 2, 4, 8, 16),
    ):
        super().__init__()
        self.image_size = int(image_size)
        self.init_size = int(init_size)
        size = self.init_size
        num_upsamples = 0
        while size < self.image_size:
            size *= 2
            num_upsamples += 1
        if size != self.image_size:
            raise ValueError('image_size must equal init_size times a power of two.')
        if len(ch_mult) < num_upsamples:
            raise ValueError(f'Need at least {num_upsamples} channel multipliers.')

        self.base_dim = int(base_dim)
        self.projection = nn.Linear(cls_dim, self.base_dim * self.init_size**2)
        channels = [max(self.base_dim // int(mult), 32) for mult in ch_mult[:num_upsamples]]
        layers = []
        input_channels = self.base_dim
        for output_channels in channels:
            layers.extend(
                [
                    nn.Upsample(scale_factor=2, mode='nearest'),
                    nn.Conv2d(input_channels, output_channels, 3, padding=1),
                    nn.GroupNorm(min(32, output_channels), output_channels),
                    nn.SiLU(),
                    nn.Conv2d(output_channels, output_channels, 3, padding=1),
                    nn.GroupNorm(min(32, output_channels), output_channels),
                    nn.SiLU(),
                ]
            )
            input_channels = output_channels
        self.upsampler = nn.Sequential(*layers)
        self.to_rgb = nn.Conv2d(input_channels, 3, 3, padding=1)

    def forward(self, cls_embedding):
        feature = self.projection(cls_embedding).reshape(
            cls_embedding.shape[0], self.base_dim, self.init_size, self.init_size
        )
        return torch.tanh(self.to_rgb(self.upsampler(feature)))


def _group_norm(channels, num_groups=32):
    return nn.GroupNorm(
        num_groups=min(num_groups, channels), num_channels=channels, eps=1e-6
    )


class _ResBlock(nn.Module):
    """Official LDM-style residual block used by CNNImageDecoder."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.norm1 = _group_norm(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = _group_norm(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, x):
        hidden = self.conv1(F.silu(self.norm1(x)))
        hidden = self.conv2(F.silu(self.norm2(hidden)))
        return hidden + self.skip(x)


class _Upsample(nn.Module):
    """Official nearest-neighbour 2x upsampling followed by a 3x3 conv."""

    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2.0, mode='nearest'))


class ConvDecoder(nn.Module):
    """Stable-pretraining CNNImageDecoder adapted to LeWM's 224px CLS token.

    This follows the official LDM/VQ-VAE-style implementation exactly.  The
    only task-specific adaptation is ``start_size=7`` because LeWM frames are
    224x224 and 224 / 7 = 32, giving five power-of-two upsampling stages.
    """

    def __init__(
        self,
        cls_dim=192,
        image_size=224,
        out_channels=3,
        base_channels=512,
        min_channels=32,
        start_size=7,
        num_res_blocks=2,
    ):
        super().__init__()
        if image_size < start_size or image_size % start_size:
            raise ValueError('image_size must be a multiple of start_size.')
        ratio = image_size // start_size
        num_stages = int(round(math.log2(ratio)))
        if 2**num_stages != ratio:
            raise ValueError('image_size / start_size must be a power of two.')
        if num_res_blocks <= 0:
            raise ValueError('num_res_blocks must be positive.')

        self.cls_dim = int(cls_dim)
        self.image_size = int(image_size)
        self.out_channels = int(out_channels)
        self.start_size = int(start_size)
        self.num_stages = num_stages
        self.channels = [
            max(int(base_channels) // (2**index), int(min_channels))
            for index in range(num_stages + 1)
        ]

        self.fc = nn.Linear(
            self.cls_dim, self.channels[0] * self.start_size * self.start_size
        )
        self.conv_in = nn.Conv2d(
            self.channels[0], self.channels[0], kernel_size=3, padding=1
        )
        stages = []
        for index in range(num_stages):
            input_channels = self.channels[index]
            output_channels = self.channels[index + 1]
            blocks = [_ResBlock(input_channels, output_channels)]
            blocks.extend(
                _ResBlock(output_channels, output_channels)
                for _ in range(num_res_blocks - 1)
            )
            blocks.append(_Upsample(output_channels))
            stages.append(nn.ModuleList(blocks))
        self.stages = nn.ModuleList(stages)
        self.norm_out = _group_norm(self.channels[-1])
        self.conv_out = nn.Conv2d(
            self.channels[-1], self.out_channels, kernel_size=3, padding=1
        )

    def forward(self, cls_embedding):
        if cls_embedding.ndim != 2 or cls_embedding.shape[1] != self.cls_dim:
            raise ValueError(
                f'Expected CLS embeddings [B, {self.cls_dim}], got '
                f'{tuple(cls_embedding.shape)}.'
            )
        hidden = self.fc(cls_embedding).reshape(
            cls_embedding.shape[0], self.channels[0], self.start_size, self.start_size
        )
        hidden = self.conv_in(hidden)
        for stage in self.stages:
            for block in stage:
                hidden = block(hidden)
        hidden = F.silu(self.norm_out(hidden))
        return self.conv_out(hidden)


def load_decoder(path, device='cpu'):
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    decoder_types = {
        'lewm_cls_visual_decoder_v1': CLSDecoder,
        'lewm_conv_visual_decoder_v1': LegacyConvDecoder,
        'lewm_official_conv_visual_decoder_v2': ConvDecoder,
    }
    if checkpoint.get('type') not in decoder_types:
        raise ValueError(f'Unsupported visual decoder checkpoint: {path}')
    model = decoder_types[checkpoint['type']](**checkpoint['model_config'])
    model.load_state_dict(checkpoint['model'], strict=True)
    return model.to(device).eval(), checkpoint
