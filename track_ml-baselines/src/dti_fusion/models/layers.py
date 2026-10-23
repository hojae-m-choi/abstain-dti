"""Attention building blocks shared by the drug encoders and the fusion transformer.

All attention goes through ``F.scaled_dot_product_attention`` with an additive float
mask, which runs on CUDA (flash/mem-efficient kernels when no bias is given), MPS and CPU.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

NEG_INF = float("-inf")


def key_padding_bias(mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """[B, Lk] bool (True = valid) -> [B, 1, 1, Lk] additive bias."""
    bias = torch.zeros(mask.shape, dtype=dtype, device=mask.device)
    bias = bias.masked_fill(~mask, NEG_INF)
    return bias[:, None, None, :]


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads, self.d_head = n_heads, d_model // n_heads
        self.q = nn.Linear(d_model, d_model)
        self.kv = nn.Linear(d_model, 2 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, context: torch.Tensor, context_mask: torch.Tensor, bias: torch.Tensor | None = None):
        b, lq, d = x.shape
        lk = context.shape[1]
        q = self.q(x).view(b, lq, self.n_heads, self.d_head).transpose(1, 2)
        k, v = self.kv(context).view(b, lk, 2, self.n_heads, self.d_head).permute(2, 0, 3, 1, 4)
        attn_bias = key_padding_bias(context_mask, q.dtype)
        if bias is not None:
            attn_bias = attn_bias + bias.to(q.dtype)
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_bias, dropout_p=self.dropout if self.training else 0.0
        )
        return self.out(y.transpose(1, 2).reshape(b, lq, d))


class FeedForward(nn.Module):
    def __init__(self, d_model: int, mult: int = 4, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, mult * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mult * d_model, d_model),
        )

    def forward(self, x):
        return self.net(x)


class SelfAttentionBlock(nn.Module):
    """Pre-LN transformer encoder block with optional additive attention bias."""

    def __init__(self, d_model: int, n_heads: int, ffn_mult: int = 4, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, ffn_mult, dropout)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask, bias=None):
        h = self.norm1(x)
        x = x + self.drop(self.attn(h, h, mask, bias))
        return x + self.drop(self.ffn(self.norm2(x)))


class CrossAttentionBlock(nn.Module):
    """Pre-LN cross-attention (queries from x, keys/values from context) + FFN."""

    def __init__(self, d_model: int, n_heads: int, ffn_mult: int = 4, dropout: float = 0.1):
        super().__init__()
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, ffn_mult, dropout)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, context, context_mask):
        x = x + self.drop(self.attn(self.norm_q(x), self.norm_kv(context), context_mask))
        return x + self.drop(self.ffn(self.norm2(x)))


class AttentionPool(nn.Module):
    """Masked attention pooling: one learned query over a token sequence."""

    def __init__(self, d_model: int):
        super().__init__()
        self.score = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))

    def forward(self, x, mask):
        logits = self.score(x).squeeze(-1).masked_fill(~mask, NEG_INF)
        weights = torch.softmax(logits.float(), dim=-1).to(x.dtype)
        return torch.einsum("bl,bld->bd", weights, x), weights
