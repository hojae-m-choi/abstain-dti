"""PyTorch dataset + collate for (drug, protein, affinity) pairs."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .featurize import TokenStore
from .graph import smiles_to_graph


class ProteinFeatures:
    """Serves ESM-2 residue tokens capped at ``max_len`` (pooled or truncated), cached in RAM."""

    def __init__(self, store: TokenStore, max_len: int, policy: str = "pool"):
        if policy not in ("pool", "truncate"):
            raise ValueError("protein_len_policy must be 'pool' or 'truncate'")
        self.store, self.max_len, self.policy = store, max_len, policy
        self.dim = store.dim
        self._cache: dict[str, torch.Tensor] = {}

    def __getitem__(self, key: str) -> torch.Tensor:
        if key not in self._cache:
            emb = torch.from_numpy(np.asarray(self.store.get(key), dtype=np.float32))
            if emb.shape[0] > self.max_len:
                if self.policy == "truncate":
                    emb = emb[: self.max_len]
                else:  # adaptive average pooling keeps coverage of the whole chain
                    emb = F.adaptive_avg_pool1d(emb.T.unsqueeze(0), self.max_len).squeeze(0).T
            self._cache[key] = emb.to(torch.float16).contiguous()
        return self._cache[key]


class GraphFeatures:
    def __init__(self, smiles_by_key: dict[str, str]):
        self.graphs = {k: smiles_to_graph(s) for k, s in smiles_by_key.items()}

    def __getitem__(self, key: str) -> dict[str, np.ndarray]:
        return self.graphs[key]


class TokenFeatures:
    """Frozen ChemBERTa token embeddings."""

    def __init__(self, store: TokenStore):
        self.store = store
        self.dim = store.dim

    def __getitem__(self, key: str) -> torch.Tensor:
        return torch.from_numpy(np.asarray(self.store.get(key), dtype=np.float32))


class SmilesText:
    """Raw SMILES for end-to-end ChemBERTa fine-tuning (tokenised in collate)."""

    def __init__(self, smiles_by_key: dict[str, str]):
        self.smiles = smiles_by_key

    def __getitem__(self, key: str) -> str:
        return self.smiles[key]


class PairDataset(Dataset):
    def __init__(self, df: pd.DataFrame, indices: np.ndarray, drug_features, protein_features: ProteinFeatures):
        part = df.iloc[indices]
        self.drug_keys = part["drug_key"].tolist()
        self.target_keys = part["target_key"].tolist()
        self.y = part["y"].to_numpy(dtype=np.float32)
        self.drug_features = drug_features
        self.protein_features = protein_features

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, i: int):
        return self.drug_features[self.drug_keys[i]], self.protein_features[self.target_keys[i]], self.y[i]


def _pad_tokens(tensors: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    lengths = [t.shape[0] for t in tensors]
    out = torch.zeros(len(tensors), max(lengths), tensors[0].shape[1], dtype=torch.float32)
    mask = torch.zeros(len(tensors), max(lengths), dtype=torch.bool)
    for i, t in enumerate(tensors):
        out[i, : lengths[i]] = t.float()
        mask[i, : lengths[i]] = True
    return out, mask


def _pad_graphs(graphs: list[dict[str, np.ndarray]]) -> dict[str, torch.Tensor]:
    n_max = max(g["atom_feat"].shape[0] for g in graphs)
    n_feat = graphs[0]["atom_feat"].shape[1]
    b = len(graphs)
    atom = torch.zeros(b, n_max, n_feat, dtype=torch.long)
    bond = torch.zeros(b, n_max, n_max, dtype=torch.long)
    spd = torch.full((b, n_max, n_max), -1, dtype=torch.long)
    mask = torch.zeros(b, n_max, dtype=torch.bool)
    for i, g in enumerate(graphs):
        n = g["atom_feat"].shape[0]
        atom[i, :n] = torch.from_numpy(g["atom_feat"])
        bond[i, :n, :n] = torch.from_numpy(g["bond"])
        spd[i, :n, :n] = torch.from_numpy(g["spd"])
        mask[i, :n] = True
    return {"atom_feat": atom, "bond": bond, "spd": spd, "mask": mask}


class Collator:
    def __init__(self, drug_mode: str, tokenizer=None, max_smiles_tokens: int = 256):
        self.drug_mode = drug_mode  # graph | tokens | smiles
        self.tokenizer = tokenizer
        self.max_smiles_tokens = max_smiles_tokens

    def __call__(self, batch):
        drugs, proteins, ys = zip(*batch)
        protein, protein_mask = _pad_tokens(list(proteins))
        out = {
            "protein": protein,
            "protein_mask": protein_mask,
            "y": torch.tensor(np.array(ys), dtype=torch.float32),
        }
        if self.drug_mode == "graph":
            out["drug"] = _pad_graphs(list(drugs))
        elif self.drug_mode == "tokens":
            emb, mask = _pad_tokens(list(drugs))
            out["drug"] = {"tokens": emb, "mask": mask}
        else:
            enc = self.tokenizer(
                list(drugs), return_tensors="pt", padding=True, truncation=True, max_length=self.max_smiles_tokens
            )
            out["drug"] = {"input_ids": enc["input_ids"], "mask": enc["attention_mask"].bool()}
        return out


def to_device(batch, device: torch.device):
    if isinstance(batch, dict):
        return {k: to_device(v, device) for k, v in batch.items()}
    if isinstance(batch, torch.Tensor):
        return batch.to(device, non_blocking=True)
    return batch
