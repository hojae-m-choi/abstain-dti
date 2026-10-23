"""ChemBERTa drug encoder.

``finetune=False`` (default): consumes cached frozen ChemBERTa token embeddings and
learns a projection plus ``adapter_layers`` transformer blocks on top.
``finetune=True``: runs ChemBERTa inside the graph and updates its weights
(with its own, smaller learning rate; see train.py).
"""

from __future__ import annotations

import torch.nn as nn

from .layers import SelfAttentionBlock


class ChemBERTaEncoder(nn.Module):
    def __init__(self, model_name: str, input_dim: int | None, d_model: int, n_heads: int,
                 adapter_layers: int = 1, dropout: float = 0.1, finetune: bool = False):
        super().__init__()
        self.finetune = finetune
        if finetune:
            from transformers import AutoModel

            self.backbone = AutoModel.from_pretrained(model_name, add_pooling_layer=False)
            input_dim = self.backbone.config.hidden_size
        self.proj = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, d_model), nn.Dropout(dropout))
        self.adapter = nn.ModuleList(
            [SelfAttentionBlock(d_model, n_heads, 4, dropout) for _ in range(adapter_layers)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.out_dim = d_model

    def forward(self, mask, tokens=None, input_ids=None):
        if self.finetune:
            tokens = self.backbone(input_ids=input_ids, attention_mask=mask.long()).last_hidden_state
        x = self.proj(tokens)
        for layer in self.adapter:
            x = layer(x, mask)
        return self.norm(x), mask
