"""Backfill finished runs into Weights & Biases.

The grid was launched without a tracker. Rather than restart 48 runs to add live
logging, this replays what each run already wrote -- config.yaml, the per-epoch
history.csv and the final metrics.json -- into one W&B run apiece. The result is the
same dashboard; only the live-while-training view is missing.

Re-running is safe: a run whose W&B id already exists is skipped unless --force.

    uv run --extra track python experiments/exp2-comparing_models1/scripts/07_to_wandb.py
    WANDB_MODE=offline uv run ... 07_to_wandb.py     # no network; sync later with `wandb sync`
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

EXP_DIR = Path(__file__).resolve().parents[1]
SPLIT_ORDER = ["cold_drug", "scaffold", "cold_target", "cold_both"]


def parse_name(name: str) -> dict:
    """n<N>_<split>_<arm>_s<seed>; split names themselves contain underscores."""
    n, rest = name[1:].split("_", 1)
    arm, seed = rest.rsplit("_s", 1)
    split = next((s for s in SPLIT_ORDER if arm.startswith(s + "_")), None)
    if split:
        arm = arm[len(split) + 1 :]
    return {"n": int(n), "split": split, "arm": arm, "seed": int(seed)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default="abstain-dti-exp2")
    ap.add_argument("--entity", default=None)
    ap.add_argument(
        "--force", action="store_true", help="re-upload runs already present"
    )
    args = ap.parse_args()

    import wandb

    done = set()
    if not args.force:
        try:
            api = wandb.Api()
            path = f"{args.entity}/{args.project}" if args.entity else args.project
            done = {r.name for r in api.runs(path)}
        except Exception as exc:  # first upload, offline mode, or no credentials
            print(f"(could not list existing runs: {exc})")

    uploaded = 0
    for metrics_path in sorted((EXP_DIR / "runs").glob("*/metrics.json")):
        run_dir = metrics_path.parent
        name = run_dir.name
        if name in done:
            print(f"skip {name} (already in W&B)")
            continue

        meta = parse_name(name)
        cfg_path = run_dir / "config.yaml"
        config = yaml.safe_load(cfg_path.read_text()) if cfg_path.exists() else {}
        results = json.loads(metrics_path.read_text())

        run = wandb.init(
            project=args.project,
            entity=args.entity,
            name=name,
            reinit=True,
            config={**config, **meta, "best_estimator": results.get("best_estimator")},
            tags=[meta["arm"], meta["split"] or "unknown", f"n{meta['n']}"],
        )
        history = run_dir / "history.csv"
        if history.exists():  # FLAML has no epochs
            for _, row in pd.read_csv(history).iterrows():
                run.log(
                    {k: v for k, v in row.items() if pd.notna(v)},
                    step=int(row["epoch"]),
                )
        run.summary.update(
            {
                **{f"test/{k}": v for k, v in results["test"].items()},
                **{f"val/{k}": v for k, v in results.get("val", {}).items()},
                "best_epoch": results.get("best_epoch"),
                "params": results.get("params"),
            }
        )
        for part in ("val", "test"):
            preds = run_dir / f"predictions_{part}.csv"
            if preds.exists():
                run.save(str(preds), base_path=str(run_dir), policy="now")
        run.finish()
        uploaded += 1
        print(f"uploaded {name}")

    print(f"\n{uploaded} run(s) uploaded to {args.project}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
