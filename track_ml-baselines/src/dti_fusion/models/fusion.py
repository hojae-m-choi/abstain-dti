"""Multimodal fusion transformer: drug tokens x protein residue tokens -> affinity.

``cross`` mode (default), per layer:
    drug    <- self-attention(drug)
    drug    <- cross-attention(query=drug,    key/value=protein) + FFN
    protein <- [optional self-attention(protein)]
    protein <- cross-attention(query=protein, key/value=drug)    + FFN
Cost is O(Ld^2 + 2*Ld*Lp) per layer, so 1,000-residue proteins fit on a laptop.

``joint`` mode: [CLS; drug; protein] with modality-type embeddings through standard
self-attention blocks, O((1 + Ld + Lp)^2). Heavier; kept as an ablation.

Readout: masked attention pooling of each modality (plus CLS in joint mode),
then an MLP on [h_d, h_p, h_d * h_p, |h_d - h_p|].
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .layers import AttentionPool, CrossAttentionBlock, SelfAttentionBlock


class CoAttentionLayer(nn.Module):
    def __init__(self, d_model, n_heads, ffn_mult, dropout, protein_self_attn):
        super().__init__()
        self.drug_self = SelfAttentionBlock(d_model, n_heads, ffn_mult, dropout)
        self.drug_cross = CrossAttentionBlock(d_model, n_heads, ffn_mult, dropout)
        self.protein_self = SelfAttentionBlock(d_model, n_heads, ffn_mult, dropout) if protein_self_attn else None
        self.protein_cross = CrossAttentionBlock(d_model, n_heads, ffn_mult, dropout)

    def forward(self, d, d_mask, p, p_mask):
        d = self.drug_self(d, d_mask)
        d_new = self.drug_cross(d, p, p_mask)
        if self.protein_self is not None:
            p = self.protein_self(p, p_mask)
        p = self.protein_cross(p, d, d_mask)  # attends to the drug state before this layer's update
        return d_new, p


class FusionTransformer(nn.Module):
    def __init__(self, d_model=256, n_layers=2, n_heads=8, ffn_mult=4, dropout=0.1,
                 mode="cross", protein_self_attn=False):
        super().__init__()
        if mode not in ("cross", "joint"):
            raise ValueError("fusion.mode must be 'cross' or 'joint'")
        self.mode = mode
        if mode == "cross":
            self.layers = nn.ModuleList(
                [CoAttentionLayer(d_model, n_heads, ffn_mult, dropout, protein_self_attn) for _ in range(n_layers)]
            )
        else:
            self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
            self.type_embedding = nn.Embedding(3, d_model)  # 0 cls, 1 drug, 2 protein
            self.layers = nn.ModuleList(
                [SelfAttentionBlock(d_model, n_heads, ffn_mult, dropout) for _ in range(n_layers)]
            )
            nn.init.normal_(self.cls, std=0.02)
        self.drug_norm = nn.LayerNorm(d_model)
        self.protein_norm = nn.LayerNorm(d_model)
        self.drug_pool = AttentionPool(d_model)
        self.protein_pool = AttentionPool(d_model)
        head_in = 4 * d_model + (d_model if mode == "joint" else 0)
        self.head = nn.Sequential(
            nn.Linear(head_in, d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Linear(d_model // 2, 1),
        )

    def forward(self, d, d_mask, p, p_mask):
        cls = None
        if self.mode == "cross":
            for layer in self.layers:
                d, p = layer(d, d_mask, p, p_mask)
        else:
            b, ld = d.shape[0], d.shape[1]
            x = torch.cat([
                self.cls.expand(b, -1, -1) + self.type_embedding.weight[0],
                d + self.type_embedding.weight[1],
                p + self.type_embedding.weight[2],
            ], dim=1)
            mask = torch.cat([d_mask.new_ones(b, 1), d_mask, p_mask], dim=1)
            for layer in self.layers:
                x = layer(x, mask)
            cls, d, p = x[:, 0], x[:, 1 : 1 + ld], x[:, 1 + ld :]
        hd, drug_attn = self.drug_pool(self.drug_norm(d), d_mask)
        hp, protein_attn = self.protein_pool(self.protein_norm(p), p_mask)
        feats = [hd, hp, hd * hp, (hd - hp).abs()] + ([cls] if cls is not None else [])
        pred = self.head(torch.cat(feats, dim=-1)).squeeze(-1)
        return pred, {"drug_attn": drug_attn, "protein_attn": protein_attn}
