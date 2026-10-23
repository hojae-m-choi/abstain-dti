"""Graphormer-style Graph Transformer for molecules, trained from scratch.

* Node input: sum of categorical atom-feature embeddings (degree acts as centrality encoding).
* Structure: per-head additive attention bias from shortest-path distance
  and from the bond type of directly bonded atom pairs.
* A virtual "graph" token attends to and from every atom with its own learned bias.

Output tokens (virtual token first, then atoms) go to the fusion transformer.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data.graph import ATOM_FEATURES, NUM_BOND_TYPES
from .layers import SelfAttentionBlock


class GraphTransformerEncoder(nn.Module):
    def __init__(self, d_model=256, n_layers=4, n_heads=8, dropout=0.1, max_dist=12):
        super().__init__()
        self.n_heads, self.max_dist = n_heads, max_dist
        self.atom_embeddings = nn.ModuleList(
            [nn.Embedding(size + 1, d_model, padding_idx=0) for _, size in ATOM_FEATURES]
        )
        self.graph_token = nn.Parameter(torch.zeros(1, 1, d_model))
        # buckets: 0..max_dist, max_dist+1 = farther, max_dist+2 = unreachable / padding
        self.spd_bias = nn.Embedding(max_dist + 3, n_heads)
        self.bond_bias = nn.Embedding(NUM_BOND_TYPES, n_heads)
        self.graph_token_bias = nn.Parameter(torch.zeros(2, n_heads))  # [to-atoms, from-atoms]
        self.layers = nn.ModuleList(
            [SelfAttentionBlock(d_model, n_heads, 4, dropout) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.out_dim = d_model
        nn.init.normal_(self.graph_token, std=0.02)
        nn.init.zeros_(self.spd_bias.weight)
        nn.init.zeros_(self.bond_bias.weight)

    def _structural_bias(self, bond, spd):
        b, n, _ = spd.shape
        bucket = torch.where(spd < 0, self.max_dist + 2, spd.clamp(max=self.max_dist + 1))
        # One-hot matmul instead of nn.Embedding lookups: the embedding backward has to
        # scatter B*N*N gradients into a ~15-row table, which is ~30x slower on MPS.
        dtype = self.spd_bias.weight.dtype
        atom_bias = (
            F.one_hot(bucket, self.spd_bias.num_embeddings).to(dtype) @ self.spd_bias.weight
            + F.one_hot(bond, self.bond_bias.num_embeddings).to(dtype) @ self.bond_bias.weight
        )  # [B, N, N, H]
        h = self.n_heads
        corner = atom_bias.new_zeros(b, 1, 1, h)
        to_atoms = self.graph_token_bias[0].view(1, 1, 1, h).expand(b, 1, n, h)
        from_atoms = self.graph_token_bias[1].view(1, 1, 1, h).expand(b, n, 1, h)
        bias = torch.cat([
            torch.cat([corner, to_atoms], dim=2),
            torch.cat([from_atoms, atom_bias], dim=2),
        ], dim=1)
        return bias.permute(0, 3, 1, 2)  # [B, H, N+1, N+1]

    def forward(self, atom_feat, bond, spd, mask):
        x = sum(emb(atom_feat[..., i]) for i, emb in enumerate(self.atom_embeddings))
        x = torch.cat([self.graph_token.expand(x.shape[0], -1, -1), x], dim=1)
        mask = torch.cat([mask.new_ones(mask.shape[0], 1), mask], dim=1)
        bias = self._structural_bias(bond, spd)
        for layer in self.layers:
            x = layer(x, mask, bias)
        return self.norm(x), mask
