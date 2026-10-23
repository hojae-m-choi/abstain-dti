"""Train/validation/test splits.

``random``      pairs shuffled independently
``cold_drug``   test drugs never seen in training
``cold_target`` test sequences never seen in training
``cold_both``   drugs and targets partitioned independently; a pair is kept only
                when both entities fall in the same partition (cross pairs dropped)
``scaffold``    Bemis-Murcko scaffold groups assigned to partitions (seeded, size-balanced)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

SPLITS = ("random", "cold_drug", "cold_target", "cold_both", "scaffold")


def _partition_entities(keys: np.ndarray, fractions, rng: np.random.Generator) -> dict[str, int]:
    unique = np.array(sorted(set(keys)))
    rng.shuffle(unique)
    n = len(unique)
    n_train = int(round(fractions[0] * n))
    n_val = int(round(fractions[1] * n))
    assignment = {}
    for i, key in enumerate(unique):
        assignment[key] = 0 if i < n_train else (1 if i < n_train + n_val else 2)
    return assignment


def _partition_groups(groups: pd.Series, fractions, rng: np.random.Generator) -> np.ndarray:
    """Greedy group assignment: shuffle groups, then fill test, val, train buckets by row count.

    Large groups are placed first (after shuffling ties) so no single scaffold overflows
    a small bucket; the seed still changes which groups land where.
    """
    sizes = groups.value_counts()
    order = list(sizes.index)
    rng.shuffle(order)
    order.sort(key=lambda g: -sizes[g])
    total = len(groups)
    caps = [fractions[0] * total, fractions[1] * total, fractions[2] * total]
    fill = [0, 0, 0]
    bucket_of = {}
    for g in order:
        # put the group in the bucket that is furthest below its quota
        deficits = [caps[b] - fill[b] for b in range(3)]
        b = int(np.argmax(deficits))
        bucket_of[g] = b
        fill[b] += sizes[g]
    return groups.map(bucket_of).to_numpy()


def murcko_scaffold(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False) or smiles
    except Exception:
        return smiles


def make_split(df: pd.DataFrame, kind: str, fractions=(0.8, 0.1, 0.1), seed: int = 0) -> dict[str, np.ndarray]:
    if kind not in SPLITS:
        raise ValueError(f"unknown split '{kind}', choose from {SPLITS}")
    if abs(sum(fractions) - 1.0) > 1e-6:
        raise ValueError("split fractions must sum to 1")
    rng = np.random.default_rng(seed)
    n = len(df)
    if kind == "random":
        perm = rng.permutation(n)
        n_train = int(round(fractions[0] * n))
        n_val = int(round(fractions[1] * n))
        bucket = np.empty(n, dtype=int)
        bucket[perm[:n_train]] = 0
        bucket[perm[n_train : n_train + n_val]] = 1
        bucket[perm[n_train + n_val :]] = 2
    elif kind in ("cold_drug", "cold_target"):
        col = "drug_key" if kind == "cold_drug" else "target_key"
        assignment = _partition_entities(df[col].to_numpy(), fractions, rng)
        bucket = df[col].map(assignment).to_numpy()
    elif kind == "cold_both":
        drug_part = df["drug_key"].map(_partition_entities(df["drug_key"].to_numpy(), fractions, rng)).to_numpy()
        target_part = df["target_key"].map(_partition_entities(df["target_key"].to_numpy(), fractions, rng)).to_numpy()
        bucket = np.where(drug_part == target_part, drug_part, -1)
    else:  # scaffold
        scaffolds = {s: murcko_scaffold(s) for s in df["smiles"].unique()}
        bucket = _partition_groups(df["smiles"].map(scaffolds), fractions, rng)
    split = {name: np.flatnonzero(bucket == b) for b, name in enumerate(("train", "val", "test"))}
    for name, idx in split.items():
        if len(idx) == 0:
            raise ValueError(f"split '{kind}' produced an empty {name} partition; adjust fractions or dataset size")
    return split


def describe_split(df: pd.DataFrame, split: dict[str, np.ndarray]) -> dict[str, dict[str, int]]:
    out = {}
    for name, idx in split.items():
        part = df.iloc[idx]
        out[name] = {
            "pairs": len(part),
            "drugs": part["drug_key"].nunique(),
            "targets": part["target_key"].nunique(),
        }
    return out
