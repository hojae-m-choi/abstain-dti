"""Build the frozen ESM-2 and ChemBERTa caches shared by all three models.

One store per model covers every size tag: a TokenStore is a key -> embedding map, so
the caches are built over the union of all sub_*.parquet and a larger --top-targets
simply rebuilds them as a superset that the smaller tags keep reading. Caches live
under a dedicated dataset name so they are never confused with a full-BindingDB store.

Each ESM-2 size gets its own store, so swapping the protein encoder is a one-axis
change that leaves every earlier run readable.

Run:  uv run python experiments/exp2-comparing_models1/scripts/02_featurize.py
      ... --esm 150M --skip-drug
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

from dti_fusion.config import ChemBERTaConfig, ProteinConfig
from dti_fusion.data.featurize import (
    build_chemberta_store,
    build_protein_store,
    chemberta_store_path,
    protein_store_path,
)
from dti_fusion.device import describe, resolve_device

EXP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP_DIR / "src"))
from naming import ESM_DEFAULT, ESM_MODELS  # noqa: E402

DATA_DIR = EXP_DIR.parents[1] / "data"
CACHE_NAME = "exp2-comparing_models1"
MAX_SMILES_TOKENS = 256


def load_frames() -> pd.DataFrame:
    paths = sorted((EXP_DIR / "data").glob("sub_*.parquet"))
    if not paths:
        raise SystemExit("no sub_*.parquet -- run 01_subsample.py first")
    return pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--esm",
        choices=tuple(ESM_MODELS),
        default=ESM_DEFAULT,
        help="protein encoder size; each size gets its own store",
    )
    ap.add_argument(
        "--skip-drug",
        action="store_true",
        help="protein store only -- the ChemBERTa cache does not depend on --esm",
    )
    args = ap.parse_args()

    print(describe(), "\n")
    device = resolve_device()
    df = load_frames()

    sequences = dict(zip(df["target_key"], df["sequence"]))
    smiles = dict(zip(df["drug_key"], df["smiles"]))
    print(
        f"{len(sequences)} targets ({sum(len(s) for s in sequences.values()):,} residues), "
        f"{len(smiles)} drugs\n"
    )

    protein_cfg = ProteinConfig(esm_model=ESM_MODELS[args.esm])
    drug_cfg = ChemBERTaConfig()

    t0 = time.time()
    build_protein_store(DATA_DIR, CACHE_NAME, sequences, protein_cfg, device)
    print(
        f"ESM-2 store ready in {time.time() - t0:.0f}s -> "
        f"{protein_store_path(DATA_DIR, CACHE_NAME, protein_cfg.esm_model)}"
    )

    if args.skip_drug:
        return 0

    t0 = time.time()
    build_chemberta_store(
        DATA_DIR, CACHE_NAME, smiles, drug_cfg, MAX_SMILES_TOKENS, device
    )
    print(
        f"ChemBERTa store ready in {time.time() - t0:.0f}s -> "
        f"{chemberta_store_path(DATA_DIR, CACHE_NAME, drug_cfg.model, MAX_SMILES_TOKENS)}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
