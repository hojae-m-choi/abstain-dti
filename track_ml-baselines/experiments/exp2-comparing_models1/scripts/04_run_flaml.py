"""FLAML arm: tabular AutoML on pooled pair features -- the no-interaction control.

The val set is handed to FLAML explicitly. Letting it build its own CV folds would
shuffle rows across the cold boundary and quietly leak unseen drugs/targets into
model selection, which is the one thing these splits exist to prevent.

Writes the same run folder shape as the neural arms so 06_summarize.py reads them alike.

Run:  uv run --extra automl python experiments/exp2-comparing_models1/scripts/04_run_flaml.py \
          --drug-feature ecfp --split cold_target --tag t300n30000 --seed 0
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from dti_fusion.data.datasets import DATASETS
from dti_fusion.data.store import TokenStore, chemberta_store_path, protein_store_path
from dti_fusion.metrics import regression_metrics

EXP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP_DIR / "src"))
from features import build_pair_matrix  # noqa: E402
from naming import run_name, split_path, subsample_path  # noqa: E402

DATA_DIR = EXP_DIR.parents[1] / "data"
CACHE_NAME = "exp2-comparing_models1"
ESM_MODEL = "facebook/esm2_t12_35M_UR50D"
CHEMBERTA_MODEL = "DeepChem/ChemBERTa-77M-MLM"
STORE_TOKENS = 256
THRESHOLD = DATASETS["bindingdb_large"].binary_threshold
ESTIMATORS = ["lgbm", "xgboost", "rf", "extra_tree"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--drug-feature", choices=("ecfp", "chemberta"), default="ecfp")
    ap.add_argument("--split", required=True)
    ap.add_argument(
        "--tag",
        required=True,
        help="size tag written by 01_subsample.py, e.g. t300n30000",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--time-budget", type=int, default=3600)
    ap.add_argument(
        "--n-jobs",
        type=int,
        default=int(os.environ.get("FLAML_N_JOBS", 4)),
        help="booster threads; left uncapped it starves the GPU arms sharing this box",
    )
    args = ap.parse_args()

    from flaml import AutoML

    df = pd.read_parquet(subsample_path(EXP_DIR, args.tag))
    split = {
        k: np.asarray(v, dtype=np.int64)
        for k, v in json.loads(
            split_path(EXP_DIR, args.tag, args.split, args.seed).read_text()
        )["indices"].items()
    }

    protein_store = TokenStore(protein_store_path(DATA_DIR, CACHE_NAME, ESM_MODEL))
    drug_store = (
        TokenStore(
            chemberta_store_path(DATA_DIR, CACHE_NAME, CHEMBERTA_MODEL, STORE_TOKENS)
        )
        if args.drug_feature == "chemberta"
        else None
    )

    name = run_name(args.tag, args.split, f"flaml-{args.drug_feature}", args.seed)
    out = EXP_DIR / "runs" / name
    out.mkdir(parents=True, exist_ok=True)
    print(f"run {name}")

    t0 = time.time()
    parts = {}
    for name, idx in split.items():
        parts[name] = build_pair_matrix(
            df.iloc[idx], protein_store, args.drug_feature, drug_store
        )
    print(f"features {parts['train'][0].shape} built in {time.time() - t0:.0f}s")

    automl = AutoML()
    automl.fit(
        X_train=parts["train"][0],
        y_train=parts["train"][1],
        X_val=parts["val"][0],
        y_val=parts["val"][1],
        task="regression",
        metric="rmse",
        eval_method="holdout",
        estimator_list=ESTIMATORS,
        time_budget=args.time_budget,
        seed=args.seed,
        verbose=1,
        n_jobs=args.n_jobs,
    )
    print(f"best estimator: {automl.best_estimator}  val rmse {automl.best_loss:.4f}")

    results = {
        "run": name,
        "device": "cpu",
        "split": args.split,
        "seed": args.seed,
        "drug_encoder": f"flaml-{args.drug_feature}",
        "best_estimator": automl.best_estimator,
        "best_config": automl.best_config,
        "feature_dim": int(parts["train"][0].shape[1]),
        "time_budget_s": args.time_budget,
        "n_jobs": args.n_jobs,
    }
    for name in ("val", "test"):
        x, y = parts[name]
        pred = automl.predict(x)
        results[name] = regression_metrics(y, pred, THRESHOLD)
        part = df.iloc[split[name]]
        pd.DataFrame(
            {
                "drug_key": part["drug_key"].values,
                "target_key": part["target_key"].values,
                "y": y,
                "pred": pred,
            }
        ).to_csv(out / f"predictions_{name}.csv", index=False)
    (out / "metrics.json").write_text(json.dumps(results, indent=2, default=str))
    print("test:", json.dumps(results["test"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
