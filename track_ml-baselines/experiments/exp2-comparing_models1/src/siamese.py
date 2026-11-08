"""Siamese baseline: the same frozen encoders as the fusion transformer, but the two
modalities only ever meet as one pooled vector each.

    ChemBERTa(frozen) -> masked mean-pool -> adapter_d ┐
                                                       ├─> [shared tower] -> hd, hp
    ESM-2(frozen)     -> masked mean-pool -> adapter_p ┘

Weight sharing sits in the tower; the adapters cannot be shared because the two
modalities arrive with different widths (384 vs 480). The prediction head is the
same module the fusion transformer uses, so the only thing this model gives up
relative to it is token-level co-attention.

Takes and returns exactly what ``DTIModel`` does, so ``dti_fusion.train`` drives it unchanged.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def masked_mean(tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """[B, L, D] -> [B, D], averaging only over real positions."""
    w = mask.unsqueeze(-1).to(tokens.dtype)
    return (tokens * w).sum(1) / w.sum(1).clamp(min=1.0)


class Adapter(nn.Module):
    """Width-matching front end, mirroring DTIModel.protein_proj."""

    def __init__(self, in_dim: int, d_model: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )

    def forward(self, x):
        return self.net(x)


class ResidualBlock(nn.Module):
    def __init__(self, d_model: int, ffn_mult: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.net = nn.Sequential(
            nn.Linear(d_model, ffn_mult * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_mult * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return x + self.net(self.norm(x))


class SiameseModel(nn.Module):
    def __init__(
        self, cfg, protein_dim: int, drug_token_dim: int, n_tower_layers: int = 2
    ):
        super().__init__()
        f = cfg.fusion
        d, dropout = f.d_model, f.dropout
        self.drug_adapter = Adapter(drug_token_dim, d, dropout)
        self.protein_adapter = Adapter(protein_dim, d, dropout)
        # the siamese part: one tower, both modalities
        self.tower = nn.Sequential(
            *[ResidualBlock(d, f.ffn_mult, dropout) for _ in range(n_tower_layers)],
            nn.LayerNorm(d),
        )
        self.head = nn.Sequential(
            nn.Linear(4 * d, d),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d, d // 2),
            nn.GELU(),
            nn.Linear(d // 2, 1),
        )

    def forward(self, batch, return_attention: bool = False):
        hd = self.tower(
            self.drug_adapter(
                masked_mean(batch["drug"]["tokens"], batch["drug"]["mask"])
            )
        )
        hp = self.tower(
            self.protein_adapter(masked_mean(batch["protein"], batch["protein_mask"]))
        )
        pred = self.head(torch.cat([hd, hp, hd * hp, (hd - hp).abs()], dim=-1)).squeeze(
            -1
        )
        # no attention to report; keep the tuple shape so callers need no special case
        return (
            (pred, {"drug_attn": None, "protein_attn": None})
            if return_attention
            else pred
        )

    def parameter_groups(self, lr: float, backbone_lr: float, weight_decay: float):
        decay, no_decay = [], []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            (no_decay if param.ndim < 2 or "norm" in name else decay).append(param)
        return [
            {"params": decay, "lr": lr, "weight_decay": weight_decay},
            {"params": no_decay, "lr": lr, "weight_decay": 0.0},
        ]
