"""LeWorldModel-compatible latent-to-RGB decoder for visualization only."""

from __future__ import annotations

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


def load_decoder(path, device='cpu'):
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    if checkpoint.get('type') != 'lewm_cls_visual_decoder_v1':
        raise ValueError(f'Unsupported visual decoder checkpoint: {path}')
    model = CLSDecoder(**checkpoint['model_config'])
    model.load_state_dict(checkpoint['model'], strict=True)
    return model.to(device).eval(), checkpoint
