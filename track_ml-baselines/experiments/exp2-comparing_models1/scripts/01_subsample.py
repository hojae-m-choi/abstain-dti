"""Build the frozen subsample and split indices that every model in this experiment reads.

Two-stage sampling (see Phase 0 findings in README.md):
  1. keep only the TOP_TARGETS busiest targets -- cuts the ESM-2 cache ~17x and makes
     cold_target dense, at the cost of fewer test targets and a well-studied-target bias;
  2. stratified pair sample within them, preserving the pAffinity decile distribution.

Writes data/sub_<n>.parquet and data/splits/<n>_<kind>_s<seed>.json so FLAML, the
Siamese model and the fusion transformer all train and score on identical rows.

Run:  uv run python experiments/exp2-comparing_models1/scripts/01_subsample.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from dti_fusion.data.datasets import DATASETS, load_dataset
from dti_fusion.data.splits import describe_split, make_split

EXP_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = EXP_DIR.parents[1] / "data"
THRESHOLD = DATASETS["bindingdb_large"].binary_threshold

TOP_TARGETS = 300
PHASE_SIZES = {"phase1": 30_000, "phase2": 100_000}
# cold_both keeps only pairs whose drug and target land in the same partition, which
# leaves ~290 test pairs at 30k -- too few to read, so it joins at phase 2.
SPLITS = {
    "phase1": ("cold_drug", "scaffold", "cold_target"),
    "phase2": ("cold_drug", "scaffold", "cold_target", "cold_both"),
}


def stratified_sample(df: pd.DataFrame, n: int, seed: int = 0) -> pd.DataFrame:
    """Sample n pairs keeping the pAffinity decile distribution."""
    if n >= len(df):
        return df.reset_index(drop=True)
    bins = pd.qcut(df["y"], q=10, labels=False, duplicates="drop")
    rng = np.random.default_rng(seed)
    frac = n / len(df)
    picked = []
    for b in np.unique(bins):
        idx = np.flatnonzero(bins == b)
        take = min(int(round(len(idx) * frac)), len(idx))
        if take:
            picked.append(rng.choice(idx, size=take, replace=False))
    return df.iloc[np.concatenate(picked)].reset_index(drop=True)


def build_subsample(df: pd.DataFrame, n: int, seed: int = 0) -> pd.DataFrame:
    busiest = (
        df.groupby("target_key").size().sort_values(ascending=False).index[:TOP_TARGETS]
    )
    return stratified_sample(df[df["target_key"].isin(set(busiest))], n, seed)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phases", default="phase1,phase2")
    ap.add_argument("--seeds", default="0,1,2")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    df = load_dataset("bindingdb_large", DATA_DIR)
    (EXP_DIR / "data" / "splits").mkdir(parents=True, exist_ok=True)
    manifest = {
        "source": "bindingdb_large",
        "top_targets": TOP_TARGETS,
        "threshold": THRESHOLD,
    }

    for phase in args.phases.split(","):
        n = PHASE_SIZES[phase]
        sub = build_subsample(df, n)
        path = EXP_DIR / "data" / f"sub_{n}.parquet"
        sub.to_parquet(path, index=False)
        residues = int(sub.drop_duplicates("target_key")["sequence"].str.len().sum())
        entry = {
            "pairs": len(sub),
            "drugs": int(sub["drug_key"].nunique()),
            "targets": int(sub["target_key"].nunique()),
            "pos_rate": float((sub["y"] >= THRESHOLD).mean()),
            "distinct_residues": residues,
            "splits": {},
        }
        print(f"\n=== {phase}: {path.name} ===")
        print(
            f"pairs {entry['pairs']}  drugs {entry['drugs']}  targets {entry['targets']}"
            f"  pos_rate {entry['pos_rate']:.3f}  residues {residues:,}"
        )

        for kind in SPLITS[phase]:
            for seed in seeds:
                split = make_split(sub, kind, (0.8, 0.1, 0.1), seed=seed)
                stats = describe_split(sub, split)
                out = EXP_DIR / "data" / "splits" / f"{n}_{kind}_s{seed}.json"
                out.write_text(
                    json.dumps(
                        {
                            "kind": kind,
                            "seed": seed,
                            "pairs_total": len(sub),
                            "stats": stats,
                            "indices": {k: v.tolist() for k, v in split.items()},
                        }
                    )
                )
                if seed == seeds[0]:
                    kept = sum(s["pairs"] for s in stats.values())
                    t = stats["test"]
                    print(
                        f"  {kind:11s} kept {kept:6d} (drop {len(sub) - kept:5d})"
                        f"  test {t['pairs']:5d} pairs / {t['drugs']:5d} drugs / {t['targets']:4d} targets"
                    )
                entry["splits"].setdefault(kind, {})[str(seed)] = stats
        manifest[phase] = entry

    (EXP_DIR / "data" / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nwrote {EXP_DIR / 'data' / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
