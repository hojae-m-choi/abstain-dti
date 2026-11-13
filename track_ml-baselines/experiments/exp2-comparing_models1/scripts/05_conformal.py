"""Split conformal prediction and selective prediction over the finished runs.

No training: every run already wrote ``predictions_val.csv`` and
``predictions_test.csv``, which is exactly what split conformal needs -- calibrate the
residual quantile on val, apply it to test.

Two questions, both asked of runs that already exist:

1. **Does the interval widen where the models are weak?** Calibrating per run gives one
   absolute-residual quantile per (split, arm, seed). If the drug splits come out
   narrow and the protein splits wide, the interval is tracking real difficulty.
   Empirical test coverage is reported alongside, because exchangeability between val
   and test is *not* guaranteed here: under cold_target the two sets hold different
   proteins, so a conformal guarantee proved for exchangeable data can fail, and the
   size of that failure is itself a result.

2. **Does abstaining buy anything?** Rank the test pairs by an uncertainty score, drop
   the worst r of them, and recompute RMSE on what is left. The score has to come from
   something already on disk, so it is the disagreement between the arms: every arm saw
   the identical test rows, so their per-pair spread is free. ``oracle`` (rank by the
   true error) and ``random`` bracket how much of the possible gain the score captures.

Run:  uv run python experiments/exp2-comparing_models1/scripts/05_conformal.py --tag t300n100000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

EXP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP_DIR / "src"))
from naming import parse_run_name  # noqa: E402

SPLIT_ORDER = ["cold_drug", "scaffold", "cold_target", "cold_both"]
ARM_ORDER = [
    "flaml-ecfp",
    "flaml-chemberta",
    "siamese-ecfp",
    "siamese",
    "fusion",
    "siamese-esm150",
    "fusion-esm150",
]
KEYS = ["drug_key", "target_key"]


def conformal_q(residuals: np.ndarray, alpha: float) -> float:
    """Split-conformal quantile of |y - pred|, with the finite-sample correction.

    The rank is ceil((n + 1)(1 - alpha)); taking the plain (1 - alpha) empirical
    quantile instead undercovers at small n, which is exactly where cold_both sits.
    """
    n = len(residuals)
    rank = int(np.ceil((n + 1) * (1 - alpha)))
    if rank > n:  # too few calibration points to promise this level at all
        return float("inf")
    return float(np.sort(residuals)[rank - 1])


def load_runs(runs_dir: Path, size_tag: str) -> dict[tuple, dict[str, pd.DataFrame]]:
    """{(split, arm, seed): {'val': df, 'test': df}} for one size tag."""
    out = {}
    for path in sorted(runs_dir.glob(f"{size_tag}_*")):
        if not (path / "predictions_test.csv").exists():
            continue
        meta = parse_run_name(path.name, SPLIT_ORDER)
        if meta["split"] is None:
            continue
        out[(meta["split"], meta["arm"], meta["seed"])] = {
            part: pd.read_csv(path / f"predictions_{part}.csv")
            for part in ("val", "test")
        }
    return out


def coverage_table(runs: dict, alpha: float) -> pd.DataFrame:
    rows = []
    for (split, arm, seed), preds in runs.items():
        val, test = preds["val"], preds["test"]
        q = conformal_q(np.abs(val["y"] - val["pred"]).to_numpy(), alpha)
        covered = (np.abs(test["y"] - test["pred"]) <= q).mean()
        rows.append(
            {
                "split": split,
                "arm": arm,
                "seed": seed,
                "n_cal": len(val),
                "n_test": len(test),
                "half_width": q,
                "coverage": covered,
                "rmse": float(np.sqrt(((test["y"] - test["pred"]) ** 2).mean())),
            }
        )
    return pd.DataFrame(rows)


def disagreement(runs: dict, split: str, seed: int) -> tuple[pd.DataFrame, list[str]]:
    """Per-pair spread across the arms that share this (split, seed) test set."""
    arms = [a for (s, a, sd) in runs if s == split and sd == seed]
    arms = [a for a in ARM_ORDER if a in arms]
    if len(arms) < 2:
        return pd.DataFrame(), []
    base = runs[(split, arms[0], seed)]["test"][KEYS + ["y"]].copy()
    for arm in arms:
        part = runs[(split, arm, seed)]["test"][KEYS + ["pred"]]
        base = base.merge(part.rename(columns={"pred": arm}), on=KEYS, how="inner")
    base["spread"] = base[arms].std(axis=1, ddof=0)
    return base, arms


def selective_table(runs: dict, rates: list[float], rng_seed: int = 0) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(rng_seed)
    pairs = sorted({(s, sd) for (s, _, sd) in runs})
    for split, seed in pairs:
        table, arms = disagreement(runs, split, seed)
        if table.empty:
            continue
        y = table["y"].to_numpy()
        for arm in arms:
            err = np.abs(table[arm].to_numpy() - y)
            scores = {
                "spread": table["spread"].to_numpy(),
                "oracle": err,
                "random": rng.random(len(err)),
            }
            for name, score in scores.items():
                for rate in rates:
                    keep = np.argsort(score)[: int(round(len(score) * (1 - rate)))]
                    rows.append(
                        {
                            "split": split,
                            "arm": arm,
                            "seed": seed,
                            "score": name,
                            "abstain": rate,
                            "kept": len(keep),
                            "rmse": float(np.sqrt((err[keep] ** 2).mean())),
                        }
                    )
    return pd.DataFrame(rows)


def ordered(series: pd.Series, order: list[str]) -> pd.Series:
    present = [v for v in order if v in set(series)]
    return pd.Categorical(series, categories=present, ordered=True)


def render(cov: pd.DataFrame, sel: pd.DataFrame, rates: list[float], alpha: float):
    lines = [f"# conformal + selective prediction (alpha = {alpha:g})", ""]

    lines += [
        f"## Split conformal — calibrated on val, applied to test ({100 * (1 - alpha):g}% target)",
        "",
        "| split | arm | seeds | half-width | coverage | test RMSE |",
        "|---|---|---|---|---|---|",
    ]
    g = cov.groupby(["split", "arm"], observed=True)
    for (split, arm), part in g:
        lines.append(
            f"| {split} | {arm} | {len(part)} | "
            f"{part['half_width'].mean():.3f} ± {part['half_width'].std(ddof=0):.3f} | "
            f"{part['coverage'].mean():.3f} ± {part['coverage'].std(ddof=0):.3f} | "
            f"{part['rmse'].mean():.3f} |"
        )

    lines += [
        "",
        "## Selective prediction — RMSE on what is kept",
        "",
        "`spread` ranks by how much the arms disagree on that pair; `oracle` ranks by the",
        "true error (the ceiling); `random` abstains blindly (the floor).",
        "",
        "| split | arm | score | "
        + " | ".join(f"@{int(r * 100)}%" for r in rates)
        + " |",
        "|---|---|---|" + "---|" * len(rates),
    ]
    sel_mean = (
        sel.groupby(["split", "arm", "score", "abstain"], observed=True)["rmse"]
        .mean()
        .reset_index()
    )
    lines += [
        "",
        "### How much of the achievable gain the disagreement score captures",
        "",
        "`(rmse@0 - spread@r) / (rmse@0 - oracle@r)`, averaged over arms. 1.0 would mean",
        "the score ranks pairs as well as knowing the true error; 0.0 means it is no",
        "better than abstaining at random.",
        "",
        "| split | " + " | ".join(f"@{int(r * 100)}%" for r in rates if r > 0) + " |",
        "|---|" + "---|" * len([r for r in rates if r > 0]),
    ]
    wide = sel_mean.pivot_table(
        index=["split", "arm", "abstain"], columns="score", values="rmse", observed=True
    ).reset_index()
    base = wide[wide["abstain"] == 0].set_index(["split", "arm"])["oracle"]
    wide["base"] = wide.set_index(["split", "arm"]).index.map(base)
    wide["captured"] = (wide["base"] - wide["spread"]) / (
        wide["base"] - wide["oracle"]
    ).replace(0, np.nan)
    for split, part in wide[wide["abstain"] > 0].groupby("split", observed=True):
        by_rate = part.groupby("abstain", observed=True)["captured"].mean()
        cells = " | ".join(f"{by_rate[r]:.0%}" for r in rates if r > 0)
        lines.append(f"| {split} | {cells} |")

    for (split, arm, score), part in sel_mean.groupby(
        ["split", "arm", "score"], observed=True
    ):
        by_rate = dict(zip(part["abstain"], part["rmse"]))
        cells = " | ".join(f"{by_rate[r]:.3f}" for r in rates)
        lines.append(f"| {split} | {arm} | {score} | {cells} |")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="t300n100000")
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--abstain", default="0.0,0.1,0.2,0.3")
    args = ap.parse_args()
    rates = [float(x) for x in args.abstain.split(",")]

    runs_dir = EXP_DIR / "runs"
    runs = load_runs(runs_dir, args.tag)
    if not runs:
        raise SystemExit(f"no runs with predictions for tag {args.tag}")
    print(f"{len(runs)} runs for {args.tag}")

    cov = coverage_table(runs, args.alpha)
    sel = selective_table(runs, rates)
    for frame in (cov, sel):
        frame["split"] = ordered(frame["split"], SPLIT_ORDER)
        frame["arm"] = ordered(frame["arm"], ARM_ORDER)
    cov = cov.sort_values(["split", "arm", "seed"])
    sel = sel.sort_values(["split", "arm", "score", "abstain"])

    text = render(cov, sel, rates, args.alpha)
    (runs_dir / f"conformal_{args.tag}.md").write_text(text)
    cov.to_csv(runs_dir / f"conformal_{args.tag}_coverage.csv", index=False)
    sel.to_csv(runs_dir / f"conformal_{args.tag}_selective.csv", index=False)
    print(text)
    print(f"wrote {runs_dir}/conformal_{args.tag}{{.md,_coverage.csv,_selective.csv}}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
