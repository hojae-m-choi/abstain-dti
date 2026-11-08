"""Phase 0: measure BindingDB before committing to a subsample size or split layout.

Reports pair / entity counts, label and length distributions, and what each cold
split actually produces -- cold_both in particular drops every pair whose drug and
target land in different partitions, so its test set can collapse.

Run:  uv run python experiments/exp2-comparing_models1/scripts/00_scout.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from dti_fusion.data.datasets import DATASETS, load_dataset
from dti_fusion.data.splits import describe_split, make_split

EXP_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = EXP_DIR.parents[1] / "data"
THRESHOLD = DATASETS["bindingdb_large"].binary_threshold
SPLITS = ("cold_drug", "cold_target", "cold_both")
SUBSAMPLES = (30_000, 100_000)


def describe_table(df: pd.DataFrame) -> dict:
    lengths = df["sequence"].str.len()
    per_target = df.groupby("target_key").size()
    per_drug = df.groupby("drug_key").size()
    return {
        "pairs": len(df),
        "drugs": int(df["drug_key"].nunique()),
        "targets": int(df["target_key"].nunique()),
        "pos_rate": float((df["y"] >= THRESHOLD).mean()),
        "y": {k: float(v) for k, v in df["y"].describe().items()},
        "seq_len": {
            "mean": float(lengths.mean()),
            "median": float(lengths.median()),
            "p90": float(lengths.quantile(0.9)),
            "max": int(lengths.max()),
            "over_1022": int((lengths > 1022).sum()),
        },
        "pairs_per_target": {
            "median": float(per_target.median()),
            "max": int(per_target.max()),
        },
        "pairs_per_drug": {
            "median": float(per_drug.median()),
            "max": int(per_drug.max()),
        },
        "total_residues": int(
            df.drop_duplicates("target_key")["sequence"].str.len().sum()
        ),
    }


def stratified_sample(df: pd.DataFrame, n: int, seed: int = 0) -> pd.DataFrame:
    """Sample n pairs keeping the pAffinity decile distribution of the full table."""
    if n >= len(df):
        return df
    bins = pd.qcut(df["y"], q=10, labels=False, duplicates="drop")
    frac = n / len(df)
    rng = np.random.default_rng(seed)
    picked = []
    for b in np.unique(bins):
        idx = np.flatnonzero(bins == b)
        take = int(round(len(idx) * frac))
        if take:
            picked.append(rng.choice(idx, size=min(take, len(idx)), replace=False))
    return df.iloc[np.concatenate(picked)].reset_index(drop=True)


def split_report(df: pd.DataFrame, seed: int = 0) -> dict:
    out = {}
    for kind in SPLITS:
        split = make_split(df, kind, (0.8, 0.1, 0.1), seed=seed)
        stats = describe_split(df, split)
        kept = sum(s["pairs"] for s in stats.values())
        out[kind] = {
            "parts": stats,
            "kept_pairs": kept,
            "dropped_pairs": len(df) - kept,
            "test_pos_rate": float((df.iloc[split["test"]]["y"] >= THRESHOLD).mean()),
        }
    return out


def main() -> int:
    t0 = time.time()
    print(
        "loading bindingdb_large (first run downloads ~1.9M rows and canonicalises SMILES)..."
    )
    df = load_dataset("bindingdb_large", DATA_DIR)
    print(f"loaded in {time.time() - t0:.0f}s\n")

    report = {"threshold": THRESHOLD, "full": describe_table(df)}
    print("=== full table ===")
    print(json.dumps(report["full"], indent=2))

    print("\n=== cold splits on the FULL table ===")
    report["full_splits"] = split_report(df)
    for kind, r in report["full_splits"].items():
        print(f"{kind}: kept {r['kept_pairs']} / dropped {r['dropped_pairs']}")
        print(f"  {r['parts']}  test pos_rate {r['test_pos_rate']:.3f}")

    report["subsamples"] = {}
    for n in SUBSAMPLES:
        sub = stratified_sample(df, n)
        entry = {"table": describe_table(sub), "splits": split_report(sub)}
        report["subsamples"][str(n)] = entry
        print(f"\n=== stratified subsample n={n} (actual {len(sub)}) ===")
        t = entry["table"]
        print(
            f"drugs {t['drugs']}  targets {t['targets']}  pos_rate {t['pos_rate']:.3f}"
            f"  distinct residues {t['total_residues']:,}"
        )
        for kind, r in entry["splits"].items():
            parts = r["parts"]
            print(
                f"  {kind}: kept {r['kept_pairs']} (dropped {r['dropped_pairs']})"
                f"  test pairs {parts['test']['pairs']}"
                f" / drugs {parts['test']['drugs']} / targets {parts['test']['targets']}"
                f"  test pos_rate {r['test_pos_rate']:.3f}"
            )

    out = EXP_DIR / "data" / "scout.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {out}  ({time.time() - t0:.0f}s total)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
