"""Full DTI model: drug encoder + ESM-2 projector + fusion transformer."""

from __future__ import annotations

import torch
import torch.nn as nn

from ..config import Config
from .chemberta import ChemBERTaEncoder
from .fusion import FusionTransformer
from .graph_transformer import GraphTransformerEncoder


class DTIModel(nn.Module):
    def __init__(self, cfg: Config, protein_dim: int, drug_token_dim: int | None = None):
        super().__init__()
        f = cfg.fusion
        self.drug_kind = cfg.drug.encoder
        if self.drug_kind == "graph":
            g = cfg.drug.graph
            self.drug_encoder = GraphTransformerEncoder(g.d_model, g.n_layers, g.n_heads, g.dropout, g.max_dist)
        elif self.drug_kind == "chemberta":
            c = cfg.drug.chemberta
            self.drug_encoder = ChemBERTaEncoder(
                c.model, drug_token_dim, f.d_model, f.n_heads, c.adapter_layers, f.dropout, c.finetune
            )
        else:
            raise ValueError(f"unknown drug encoder '{self.drug_kind}' (graph | chemberta)")
        self.drug_proj = (
            nn.Identity() if self.drug_encoder.out_dim == f.d_model else nn.Linear(self.drug_encoder.out_dim, f.d_model)
        )
        self.protein_proj = nn.Sequential(
            nn.LayerNorm(protein_dim), nn.Linear(protein_dim, f.d_model), nn.GELU(), nn.Dropout(f.dropout),
            nn.Linear(f.d_model, f.d_model),
        )
        self.fusion = FusionTransformer(
            f.d_model, f.n_layers, f.n_heads, f.ffn_mult, f.dropout, f.mode, f.protein_self_attn
        )

    def encode_drug(self, drug: dict[str, torch.Tensor]):
        if self.drug_kind == "graph":
            return self.drug_encoder(drug["atom_feat"], drug["bond"], drug["spd"], drug["mask"])
        return self.drug_encoder(drug["mask"], tokens=drug.get("tokens"), input_ids=drug.get("input_ids"))

    def forward(self, batch, return_attention: bool = False):
        d, d_mask = self.encode_drug(batch["drug"])
        p = self.protein_proj(batch["protein"])
        pred, attn = self.fusion(self.drug_proj(d), d_mask, p, batch["protein_mask"])
        return (pred, attn) if return_attention else pred

    def parameter_groups(self, lr: float, backbone_lr: float, weight_decay: float):
        """Separate (smaller) LR for a fine-tuned pretrained backbone; no decay on norms/biases."""
        groups = {"decay": [], "no_decay": [], "backbone": []}
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith("drug_encoder.backbone."):
                groups["backbone"].append(param)
            elif param.ndim < 2 or "norm" in name or "embedding" in name or name.endswith("_token"):
                groups["no_decay"].append(param)
            else:
                groups["decay"].append(param)
        out = [
            {"params": groups["decay"], "lr": lr, "weight_decay": weight_decay},
            {"params": groups["no_decay"], "lr": lr, "weight_decay": 0.0},
        ]
        if groups["backbone"]:
            out.append({"params": groups["backbone"], "lr": backbone_lr, "weight_decay": weight_decay})
        return out
