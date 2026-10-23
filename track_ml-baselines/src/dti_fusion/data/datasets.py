"""Hugging Face dataset registry and loading.

Every dataset is normalised to one table with columns
``smiles, sequence, y, drug_key, target_key`` where duplicate
(canonical SMILES, sequence) pairs are averaged so a pair can never
appear on both sides of a split.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")


@dataclass(frozen=True)
class DatasetSpec:
    repo: str
    filename: str
    smiles_col: str
    sequence_col: str
    label_col: str
    label_name: str
    binary_threshold: float  # y >= threshold counts as a binder for AUROC/AUPRC
    license: str
    note: str = ""


DATASETS: dict[str, DatasetSpec] = {
    "davis": DatasetSpec(
        repo="amirhallaji/davis",
        filename="davis.csv",
        smiles_col="Molecule Sequence",
        sequence_col="Protein Sequence",
        label_col="Binding Affinity",
        label_name="pKd",
        binary_threshold=7.0,  # DeepDTA/GraphDTA convention for DAVIS
        license="MIT",
        note="68 kinase inhibitors x 442 kinases; ~70% of labels are the censored floor pKd=5.0",
    ),
    "kiba": DatasetSpec(
        repo="amirhallaji/kiba",
        filename="kiba.csv",
        smiles_col="Molecule Sequence",
        sequence_col="Protein Sequence",
        label_col="Binding Affinity",
        label_name="KIBA score",
        binary_threshold=12.1,  # DeepDTA/GraphDTA convention for KIBA
        license="Apache-2.0",
        note="2,068 compounds x 229 kinases; KIBA integrates Ki, Kd and IC50",
    ),
    "bindingdb_large": DatasetSpec(
        repo="jglaser/binding_affinity",
        filename="data/all.parquet",
        smiles_col="smiles_can",
        sequence_col="seq",
        label_col="neg_log10_affinity_M",
        label_name="pAffinity",
        binary_threshold=6.0,  # project decision: binds <=> pAffinity >= 6 (EVALUATION.md 1.3)
        license="mixed (BindingDB CC BY / PDBbind-cn / BioLiP / MOAD); do not redistribute",
        note="1.9M pairs; contains PDBbind-derived rows, so only for internal pre-training/validation",
    ),
}


def stable_key(text: str) -> str:
    return hashlib.sha1(text.encode()).hexdigest()[:16]


def canonical_smiles(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)


def download(name: str, data_dir: str | Path) -> Path:
    from huggingface_hub import hf_hub_download

    spec = DATASETS[name]
    target = Path(data_dir) / "raw" / name
    return Path(
        hf_hub_download(spec.repo, spec.filename, repo_type="dataset", local_dir=str(target))
    )


def load_dataset(name: str, data_dir: str | Path, subsample: int | None = None, seed: int = 0) -> pd.DataFrame:
    """Return the cleaned pair table, caching it as parquet under ``data_dir/processed``."""
    spec = DATASETS[name]
    processed = Path(data_dir) / "processed" / f"{name}.parquet"
    if processed.exists():
        df = pd.read_parquet(processed)
    else:
        raw_path = download(name, data_dir)
        raw = pd.read_parquet(raw_path) if raw_path.suffix == ".parquet" else pd.read_csv(raw_path)
        df = pd.DataFrame(
            {
                "smiles": raw[spec.smiles_col].astype(str),
                "sequence": raw[spec.sequence_col].astype(str).str.upper().str.strip(),
                "y": raw[spec.label_col].astype(float),
            }
        )
        df = df.dropna(subset=["y"])
        df = df[np.isfinite(df["y"])]
        canon = {s: canonical_smiles(s) for s in df["smiles"].unique()}
        df["smiles"] = df["smiles"].map(canon)
        n_bad = int(df["smiles"].isna().sum())
        df = df.dropna(subset=["smiles"])
        n_rows = len(df)
        df = df.groupby(["smiles", "sequence"], as_index=False)["y"].mean()
        df["drug_key"] = df["smiles"].map(stable_key)
        df["target_key"] = df["sequence"].map(stable_key)
        processed.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(processed, index=False)
        print(
            f"[{name}] {n_rows} rows -> {len(df)} unique pairs "
            f"({n_bad} unparseable SMILES dropped, duplicates averaged)"
        )
    if subsample is not None and subsample < len(df):
        df = df.sample(n=subsample, random_state=seed).reset_index(drop=True)
    return df.reset_index(drop=True)
