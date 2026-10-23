"""Drug-encoder comparison: ChemBERTa vs Graph Transformer over splits x seeds.

Each cell is a normal ``train`` run (so every run directory is self-contained);
this module only builds the grid and aggregates ``metrics.json`` files.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pandas as pd

from .config import Config
from .train import default_run_name, train

ENCODERS = {
    "graph": {"drug.encoder": "graph"},
    "chemberta-frozen": {"drug.encoder": "chemberta", "drug.chemberta.finetune": False},
    "chemberta-ft": {"drug.encoder": "chemberta", "drug.chemberta.finetune": True},
}
METRICS = ("rmse", "mse", "pearson", "spearman", "ci", "auroc", "auprc")


def _set(cfg: Config, dotted: str, value) -> None:
    node = cfg
    *path, leaf = dotted.split(".")
    for part in path:
        node = getattr(node, part)
    setattr(node, leaf, value)


def run_grid(base: Config, encoders: list[str], splits: list[str], seeds: list[int], skip_existing: bool = True):
    rows = []
    for split in splits:
        for seed in seeds:
            for enc in encoders:
                cfg = copy.deepcopy(base)
                for key, value in ENCODERS[enc].items():
                    _set(cfg, key, value)
                cfg.data.split, cfg.data.split_seed, cfg.train.seed = split, seed, seed
                cfg.train.run_name = None
                metrics_path = Path(cfg.train.out_dir) / default_run_name(cfg) / "metrics.json"
                if skip_existing and metrics_path.exists():
                    result = json.loads(metrics_path.read_text())
                    print(f"skip existing {metrics_path.parent.name}")
                else:
                    result = train(cfg)
                rows.append({"encoder": enc, "split": split, "seed": seed,
                             **{m: result["test"].get(m) for m in METRICS}})
    return summarize(pd.DataFrame(rows), Path(base.train.out_dir))


def summarize(df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "compare_runs.csv", index=False)
    agg = df.groupby(["split", "encoder"])[list(METRICS)].agg(["mean", "std"])
    agg.to_csv(out_dir / "compare_summary.csv")
    lines = ["| split | encoder | " + " | ".join(METRICS) + " |", "|" + "---|" * (len(METRICS) + 2)]
    for (split, enc), row in agg.iterrows():
        cells = []
        for m in METRICS:
            mean, std = row[(m, "mean")], row[(m, "std")]
            cells.append("-" if pd.isna(mean) else f"{mean:.3f} ± {0 if pd.isna(std) else std:.3f}")
        lines.append(f"| {split} | {enc} | " + " | ".join(cells) + " |")
    table = "\n".join(lines)
    (out_dir / "compare_summary.md").write_text(table + "\n")
    print(table)
    return agg
