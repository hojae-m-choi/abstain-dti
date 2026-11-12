"""Train the fusion transformer or the siamese model on a frozen subsample + split.

Both arms read the identical rows, the identical ESM-2 / ChemBERTa caches and the
identical training loop (`dti_fusion.train.train`); only the model class differs.

Run:  uv run python experiments/exp2-comparing_models1/scripts/03_run_neural.py \
          --model siamese --split cold_target --tag t300n30000 --seed 0
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from dti_fusion.config import load_config
from dti_fusion.data.dataset import (
    Collator,
    PairDataset,
    ProteinFeatures,
    TokenFeatures,
)
from dti_fusion.data.featurize import (
    TokenStore,
    chemberta_store_path,
    protein_store_path,
)
from dti_fusion.device import resolve_device
from dti_fusion.train import Prepared, train

EXP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP_DIR / "src"))
from naming import run_name, split_path, subsample_path  # noqa: E402
from siamese import SiameseModel  # noqa: E402

DATA_DIR = EXP_DIR.parents[1] / "data"
CACHE_NAME = "exp2-comparing_models1"
MAX_SMILES_TOKENS = 256


def load_split(size_tag: str, kind: str, seed: int) -> dict[str, np.ndarray]:
    meta = json.loads(split_path(EXP_DIR, size_tag, kind, seed).read_text())
    return {k: np.asarray(v, dtype=np.int64) for k, v in meta["indices"].items()}


def build_prepared(cfg, device, size_tag: str, kind: str, seed: int) -> Prepared:
    df = pd.read_parquet(subsample_path(EXP_DIR, size_tag))
    split = load_split(size_tag, kind, seed)

    protein_store = TokenStore(
        protein_store_path(DATA_DIR, CACHE_NAME, cfg.protein.esm_model)
    )
    drug_store = TokenStore(
        chemberta_store_path(
            DATA_DIR, CACHE_NAME, cfg.drug.chemberta.model, MAX_SMILES_TOKENS
        )
    )
    proteins = ProteinFeatures(
        protein_store, cfg.data.max_protein_len, cfg.data.protein_len_policy
    )
    # the store holds up to MAX_SMILES_TOKENS; serve a tighter cap so one 256-token
    # outlier cannot set the padded length -- and therefore the attention cost -- of its batch
    drugs = TokenFeatures(
        drug_store, cfg.data.max_smiles_tokens, cfg.data.protein_len_policy
    )
    collate = Collator("tokens")

    loaders = {
        name: DataLoader(
            PairDataset(df, idx, drugs, proteins),
            batch_size=cfg.train.batch_size,
            shuffle=(name == "train"),
            collate_fn=collate,
            num_workers=cfg.train.num_workers,
            pin_memory=device.type == "cuda",
        )
        for name, idx in split.items()
    }
    return Prepared(df, split, loaders, protein_store.dim, drug_store.dim)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=("fusion", "siamese"), required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument(
        "--tag",
        required=True,
        help="size tag written by 01_subsample.py, e.g. t300n30000",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--set", "-s", action="append", default=[], metavar="KEY=VALUE")
    args = ap.parse_args()

    overrides = [
        "data.dataset=bindingdb_large",  # only used for the pAffinity>=6.0 threshold
        f"data.split={args.split}",
        f"train.seed={args.seed}",
        "drug.encoder=chemberta",
        "drug.chemberta.finetune=false",
        f"train.num_workers={os.environ.get('DTI_NUM_WORKERS', 0)}",
        *args.set,
    ]
    cfg = load_config(EXP_DIR / "config.yaml", overrides)
    cfg.train.out_dir = str(EXP_DIR / "runs")
    cfg.train.run_name = run_name(args.tag, args.split, args.model, args.seed)

    device = resolve_device(cfg.train.device)
    prep = build_prepared(cfg, device, args.tag, args.split, args.seed)
    factory = SiameseModel if args.model == "siamese" else None
    train(cfg, prep=prep, build_model=factory)
    return 0


if __name__ == "__main__":
    sys.exit(main())
