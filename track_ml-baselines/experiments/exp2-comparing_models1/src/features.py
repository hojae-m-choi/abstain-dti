"""Fixed-length pair features for the FLAML arm.

Two drug representations, so that "AutoML is weaker" can be told apart from
"ECFP4 is weaker than ChemBERTa":

    ecfp      ECFP4 2048 bits          + ESM-2 mean  (2528 dims)
    chemberta ChemBERTa token mean 384 + ESM-2 mean  ( 864 dims)

The protein side is the same mean-pooled ESM-2 vector the neural models see, so the
only thing FLAML lacks is a way to model the interaction between the two sides.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator

from dti_fusion.data.store import TokenStore

RDLogger.DisableLog("rdApp.*")

ECFP_RADIUS = 2  # ECFP4
ECFP_BITS = 2048
DRUG_FEATURES = ("ecfp", "chemberta")


def ecfp_matrix(smiles_by_key: dict[str, str]) -> tuple[list[str], np.ndarray]:
    gen = rdFingerprintGenerator.GetMorganGenerator(
        radius=ECFP_RADIUS, fpSize=ECFP_BITS
    )
    keys = list(smiles_by_key)
    out = np.zeros((len(keys), ECFP_BITS), dtype=np.float32)
    for i, key in enumerate(keys):
        mol = Chem.MolFromSmiles(smiles_by_key[key])
        if mol is not None:
            out[i] = np.asarray(gen.GetFingerprintAsNumPy(mol), dtype=np.float32)
    return keys, out


def store_means(store: TokenStore, keys: list[str]) -> np.ndarray:
    """Mean over the token axis -- the same pooling the siamese model applies."""
    out = np.zeros((len(keys), store.dim), dtype=np.float32)
    for i, key in enumerate(keys):
        out[i] = np.asarray(store.get(key), dtype=np.float32).mean(axis=0)
    return out


def _lookup(table: np.ndarray, keys: list[str], wanted: pd.Series) -> np.ndarray:
    pos = {k: i for i, k in enumerate(keys)}
    return table[
        np.fromiter((pos[k] for k in wanted), dtype=np.int64, count=len(wanted))
    ]


def build_pair_matrix(
    df: pd.DataFrame,
    protein_store: TokenStore,
    drug_feature: str,
    drug_store: TokenStore | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (X, y) with one row per pair of ``df``, in ``df`` order."""
    if drug_feature not in DRUG_FEATURES:
        raise ValueError(f"drug_feature must be one of {DRUG_FEATURES}")

    smiles_by_key = dict(zip(df["drug_key"], df["smiles"]))
    if drug_feature == "ecfp":
        drug_keys, drug_table = ecfp_matrix(smiles_by_key)
    else:
        if drug_store is None:
            raise ValueError("drug_feature='chemberta' needs a drug_store")
        drug_keys = list(smiles_by_key)
        drug_table = store_means(drug_store, drug_keys)

    target_keys = list(dict.fromkeys(df["target_key"]))
    protein_table = store_means(protein_store, target_keys)

    x = np.hstack(
        [
            _lookup(drug_table, drug_keys, df["drug_key"]),
            _lookup(protein_table, target_keys, df["target_key"]),
        ]
    )
    return x, df["y"].to_numpy(dtype=np.float32)
