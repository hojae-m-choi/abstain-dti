"""Collect every finished run into one table: per-run rows and a mean +- std summary.

Run:  uv run python experiments/exp2-comparing_models1/scripts/06_summarize.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

EXP_DIR = Path(__file__).resolve().parents[1]
METRICS = ("rmse", "mae", "pearson", "spearman", "ci", "auroc", "auprc")
ARM_ORDER = ["flaml-ecfp", "flaml-chemberta", "siamese", "fusion"]
SPLIT_ORDER = ["cold_drug", "scaffold", "cold_target", "cold_both"]


def rows() -> pd.DataFrame:
    out = []
    for path in sorted((EXP_DIR / "runs").glob("*/metrics.json")):
        m = json.loads(path.read_text())
        # run names are n<N>_<split>_<arm>_s<seed>; split names contain underscores
        name = path.parent.name
        n, rest = name[1:].split("_", 1)
        arm, seed = rest.rsplit("_s", 1)
        for split in SPLIT_ORDER:
            if arm.startswith(split + "_"):
                arm = arm[len(split) + 1 :]
                break
        else:
            split = None
        out.append(
            {
                "n": int(n),
                "split": split,
                "arm": arm,
                "seed": int(seed),
                "best_epoch": m.get("best_epoch"),
                "params": m.get("params"),
                "best_estimator": m.get("best_estimator"),
                **{k: m["test"][k] for k in METRICS},
            }
        )
    return pd.DataFrame(out)


def ordered(df: pd.DataFrame, col: str, order: list[str]) -> pd.Categorical:
    extra = [v for v in df[col].dropna().unique() if v not in order]
    return pd.Categorical(df[col], categories=order + extra, ordered=True)


def main() -> int:
    df = rows()
    if df.empty:
        print("no finished runs yet")
        return 0
    df["split"] = ordered(df, "split", SPLIT_ORDER)
    df["arm"] = ordered(df, "arm", ARM_ORDER)
    df = df.sort_values(["n", "split", "arm", "seed"])
    df.to_csv(EXP_DIR / "runs" / "all_runs.csv", index=False)

    grouped = df.groupby(["n", "split", "arm"], observed=True)[list(METRICS)]
    summary = grouped.agg(["mean", "std", "count"])
    summary.to_csv(EXP_DIR / "runs" / "summary.csv")

    lines = [f"# exp2-comparing_models1 — {len(df)} runs\n"]
    for n, block in df.groupby("n"):
        lines.append(f"\n## n = {n}\n")
        lines.append(
            "| split | arm | seeds | " + " | ".join(m.upper() for m in METRICS) + " |"
        )
        lines.append("|---" * (3 + len(METRICS)) + "|")
        for (split, arm), g in block.groupby(["split", "arm"], observed=True):
            cells = []
            for m in METRICS:
                mean = g[m].mean()
                cells.append(
                    f"{mean:.3f}" if len(g) < 2 else f"{mean:.3f} ± {g[m].std():.3f}"
                )
            lines.append(f"| {split} | {arm} | {len(g)} | " + " | ".join(cells) + " |")
    text = "\n".join(lines) + "\n"
    (EXP_DIR / "runs" / "summary.md").write_text(text)
    print(text)
    print(f"wrote {EXP_DIR / 'runs'}/{{all_runs.csv,summary.csv,summary.md}}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
