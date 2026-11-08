"""Frozen feature extraction with on-disk caching.

ESM-2 residue embeddings and ChemBERTa token embeddings are computed once per
(dataset, model) and kept in a ``TokenStore``, which is memory-mapped afterwards.
The store format itself lives in ``store.py`` and stays torch-free; this module
holds the extractors that need a model.
"""

from __future__ import annotations

import numpy as np
import torch
from tqdm import tqdm

from ..device import inference_dtype
from .store import TokenStore, _slug, chemberta_store_path, protein_store_path

__all__ = [
    "TokenStore", "chemberta_store_path", "protein_store_path",
    "embed_proteins", "embed_smiles", "build_protein_store", "build_chemberta_store",
]


def _quiet_transformers() -> None:
    from transformers.utils import logging

    logging.set_verbosity_error()  # hides the "UNEXPECTED lm_head" load report for encoder-only use


# --------------------------------------------------------------------------- ESM-2


def _window_starts(length: int, window: int, overlap: int) -> list[int]:
    if length <= window:
        return [0]
    step = window - overlap
    starts = list(range(0, length - window, step))
    starts.append(length - window)  # last window flush with the C-terminus
    return starts


@torch.no_grad()
def embed_proteins(
    sequences: dict[str, str],
    model_name: str,
    device: torch.device,
    window: int = 1022,
    overlap: int = 256,
    tokens_per_batch: int = 8192,
) -> dict[str, np.ndarray]:
    """Per-residue ESM-2 embeddings (BOS/EOS stripped) for every sequence.

    Chains longer than ``window`` are embedded in overlapping windows and averaged
    where windows overlap, so every residue gets an embedding.
    """
    from transformers import AutoTokenizer, EsmModel

    _quiet_transformers()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = EsmModel.from_pretrained(model_name, add_pooling_layer=False, dtype=inference_dtype(device))
    model = model.to(device).eval()

    # flatten into (key, start, fragment) jobs, then batch by token budget
    jobs = []
    for key, seq in sequences.items():
        for start in _window_starts(len(seq), window, overlap):
            jobs.append((key, start, seq[start : start + window]))
    jobs.sort(key=lambda j: len(j[2]))

    sums = {k: np.zeros((len(s), model.config.hidden_size), dtype=np.float32) for k, s in sequences.items()}
    counts = {k: np.zeros((len(s), 1), dtype=np.float32) for k, s in sequences.items()}

    batches, current, longest = [], [], 0
    for job in jobs:
        longest_next = max(longest, len(job[2]) + 2)
        if current and longest_next * (len(current) + 1) > tokens_per_batch:
            batches.append(current)
            current, longest_next = [], len(job[2]) + 2
        current.append(job)
        longest = longest_next
    if current:
        batches.append(current)

    for batch in tqdm(batches, desc=f"ESM-2 {model_name.split('/')[-1]}"):
        enc = tokenizer([j[2] for j in batch], return_tensors="pt", padding=True)
        enc = {k: v.to(device) for k, v in enc.items()}
        hidden = model(**enc).last_hidden_state.float().cpu().numpy()
        for row, (key, start, frag) in enumerate(batch):
            emb = hidden[row, 1 : len(frag) + 1]
            sums[key][start : start + len(frag)] += emb
            counts[key][start : start + len(frag)] += 1
    del model
    if device.type == "mps":
        torch.mps.empty_cache()
    return {k: sums[k] / counts[k] for k in sequences}


def build_protein_store(data_dir, dataset, sequences: dict[str, str], cfg, device) -> TokenStore:
    root = protein_store_path(data_dir, dataset, cfg.esm_model)
    if TokenStore.exists(root):
        store = TokenStore(root)
        if all(k in store for k in sequences):
            return store
    items = embed_proteins(sequences, cfg.esm_model, device, cfg.window, cfg.window_overlap, cfg.tokens_per_batch)
    return TokenStore.write(root, items, extra={"model": cfg.esm_model})


# --------------------------------------------------------------------------- ChemBERTa


@torch.no_grad()
def embed_smiles(
    smiles: dict[str, str], model_name: str, device: torch.device, max_tokens: int = 256, batch_size: int = 128
) -> dict[str, np.ndarray]:
    """Last-layer ChemBERTa token embeddings (including <s>/</s>) per molecule."""
    from transformers import AutoModel, AutoTokenizer

    _quiet_transformers()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, add_pooling_layer=False, dtype=inference_dtype(device))
    model = model.to(device).eval()
    max_tokens = min(max_tokens, model.config.max_position_embeddings - 2)
    keys = sorted(smiles, key=lambda k: len(smiles[k]))
    out = {}
    for i in tqdm(range(0, len(keys), batch_size), desc=f"ChemBERTa {model_name.split('/')[-1]}"):
        chunk = keys[i : i + batch_size]
        enc = tokenizer(
            [smiles[k] for k in chunk], return_tensors="pt", padding=True, truncation=True, max_length=max_tokens
        )
        lengths = enc["attention_mask"].sum(1).tolist()
        enc = {k: v.to(device) for k, v in enc.items()}
        hidden = model(**enc).last_hidden_state.float().cpu().numpy()
        for row, key in enumerate(chunk):
            out[key] = hidden[row, : lengths[row]]
    return out


def build_chemberta_store(data_dir, dataset, smiles: dict[str, str], cfg, max_tokens, device) -> TokenStore:
    root = chemberta_store_path(data_dir, dataset, cfg.model, max_tokens)
    if TokenStore.exists(root):
        store = TokenStore(root)
        if all(k in store for k in smiles):
            return store
    items = embed_smiles(smiles, cfg.model, device, max_tokens)
    return TokenStore.write(root, items, extra={"model": cfg.model})
