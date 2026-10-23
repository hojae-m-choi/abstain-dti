"""Training / evaluation entry point.

Writes to ``<out_dir>/<run_name>/``:
  config.yaml, split.json, history.csv, best.pt, metrics.json, predictions_{val,test}.csv
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from .config import Config
from .data.dataset import (
    Collator, GraphFeatures, PairDataset, ProteinFeatures, SmilesText, TokenFeatures, to_device,
)
from .data.datasets import DATASETS, load_dataset
from .data.featurize import build_chemberta_store, build_protein_store
from .data.splits import describe_split, make_split
from .device import autocast, configure_backends, resolve_device, seed_everything
from .metrics import regression_metrics
from .models.model import DTIModel


def default_run_name(cfg: Config) -> str:
    drug = cfg.drug.encoder
    if drug == "chemberta":
        drug += "-ft" if cfg.drug.chemberta.finetune else "-frozen"
    esm = cfg.protein.esm_model.split("/")[-1].split("_")[1]
    return f"{cfg.data.dataset}_{cfg.data.split}_{drug}_{esm}_{cfg.fusion.mode}_s{cfg.train.seed}"


@dataclass
class Prepared:
    df: pd.DataFrame
    split: dict[str, np.ndarray]
    loaders: dict[str, DataLoader]
    protein_dim: int
    drug_token_dim: int | None


def prepare(cfg: Config, device: torch.device) -> Prepared:
    d = cfg.data
    df = load_dataset(d.dataset, d.data_dir, d.subsample, seed=d.split_seed)
    split = make_split(df, d.split, d.fractions, d.split_seed)

    sequences = dict(zip(df["target_key"], df["sequence"]))
    smiles = dict(zip(df["drug_key"], df["smiles"]))
    protein_store = build_protein_store(d.data_dir, d.dataset, sequences, cfg.protein, device)
    proteins = ProteinFeatures(protein_store, d.max_protein_len, d.protein_len_policy)

    tokenizer, drug_token_dim = None, None
    if cfg.drug.encoder == "graph":
        drugs, mode = GraphFeatures(smiles), "graph"
    elif cfg.drug.chemberta.finetune:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(cfg.drug.chemberta.model)
        drugs, mode = SmilesText(smiles), "smiles"
    else:
        store = build_chemberta_store(d.data_dir, d.dataset, smiles, cfg.drug.chemberta, d.max_smiles_tokens, device)
        drugs, mode, drug_token_dim = TokenFeatures(store), "tokens", store.dim

    collate = Collator(mode, tokenizer, d.max_smiles_tokens)
    loaders = {}
    for name, idx in split.items():
        ds = PairDataset(df, idx, drugs, proteins)
        loaders[name] = DataLoader(
            ds,
            batch_size=cfg.train.batch_size,
            shuffle=(name == "train"),
            collate_fn=collate,
            num_workers=cfg.train.num_workers,
            pin_memory=device.type == "cuda",
            drop_last=False,
        )
    return Prepared(df, split, loaders, protein_store.dim, drug_token_dim)


def _loss(pred, target, kind):
    if kind == "huber":
        return F.smooth_l1_loss(pred, target, beta=1.0)
    return F.mse_loss(pred, target)


@torch.no_grad()
def predict(model, loader, device, amp, y_mean, y_std):
    model.eval()
    preds, ys = [], []
    for batch in loader:
        batch = to_device(batch, device)
        with autocast(device, amp):
            out = model(batch)
        preds.append((out.float() * y_std + y_mean).cpu().numpy())
        ys.append(batch["y"].cpu().numpy())
    return np.concatenate(ys), np.concatenate(preds)


def train(cfg: Config) -> dict:
    configure_backends()
    seed_everything(cfg.train.seed)
    device = resolve_device(cfg.train.device)
    run_name = cfg.train.run_name or default_run_name(cfg)
    out = Path(cfg.train.out_dir) / run_name
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False, allow_unicode=True))
    print(f"run {run_name} on {device}")

    prep = prepare(cfg, device)
    spec = DATASETS[cfg.data.dataset]
    split_info = describe_split(prep.df, prep.split)
    (out / "split.json").write_text(json.dumps(split_info, indent=2))
    print("split:", split_info)

    y_train = prep.df["y"].to_numpy()[prep.split["train"]]
    y_mean, y_std = float(y_train.mean()), float(y_train.std() + 1e-8)

    seed_everything(cfg.train.seed)  # model init independent of featurisation randomness
    model = DTIModel(cfg, prep.protein_dim, prep.drug_token_dim).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable parameters: {n_params / 1e6:.2f}M")

    t = cfg.train
    optimizer = torch.optim.AdamW(model.parameter_groups(t.lr, cfg.drug.chemberta.lr, t.weight_decay))
    steps = t.epochs * len(prep.loaders["train"])
    warmup = max(1, int(t.warmup_frac * steps))
    base_lrs = [g["lr"] for g in optimizer.param_groups]

    def set_lr(step):
        scale = step / warmup if step < warmup else 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, steps - warmup)))
        for g, lr in zip(optimizer.param_groups, base_lrs):
            g["lr"] = lr * scale

    history, best, best_epoch, step = [], math.inf, -1, 0
    for epoch in range(1, t.epochs + 1):
        model.train()
        start, running, seen = time.time(), 0.0, 0
        for batch in prep.loaders["train"]:
            batch = to_device(batch, device)
            set_lr(step)
            with autocast(device, t.amp):
                pred = model(batch)
            loss = _loss(pred.float(), (batch["y"] - y_mean) / y_std, t.loss)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if t.grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), t.grad_clip)
            optimizer.step()
            step += 1
            running += loss.item() * len(pred)
            seen += len(pred)
        yv, pv = predict(model, prep.loaders["val"], device, t.amp, y_mean, y_std)
        val = regression_metrics(yv, pv, spec.binary_threshold)
        row = {"epoch": epoch, "train_loss": running / seen, "time_s": time.time() - start,
               **{f"val_{k}": v for k, v in val.items()}}
        history.append(row)
        pd.DataFrame(history).to_csv(out / "history.csv", index=False)
        score = val[t.select_metric]
        improved = score < best
        if improved:
            best, best_epoch = score, epoch
            torch.save({"model": model.state_dict(), "y_mean": y_mean, "y_std": y_std,
                        "config": cfg.to_dict(), "epoch": epoch}, out / "best.pt")
        print(f"epoch {epoch:3d}  loss {row['train_loss']:.4f}  val rmse {val['rmse']:.4f}  "
              f"ci {val['ci']:.4f}  pearson {val['pearson']:.4f}  {row['time_s']:.1f}s{'  *' if improved else ''}")
        if epoch - best_epoch >= t.patience:
            print(f"early stop at epoch {epoch} (best {best_epoch})")
            break

    state = torch.load(out / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    results = {"run": run_name, "best_epoch": best_epoch, "device": str(device), "params": n_params,
               "dataset": cfg.data.dataset, "split": cfg.data.split, "drug_encoder": cfg.drug.encoder}
    for name in ("val", "test"):
        y, p = predict(model, prep.loaders[name], device, t.amp, y_mean, y_std)
        results[name] = regression_metrics(y, p, spec.binary_threshold)
        part = prep.df.iloc[prep.split[name]]
        pd.DataFrame({"drug_key": part["drug_key"].values, "target_key": part["target_key"].values,
                      "y": y, "pred": p}).to_csv(out / f"predictions_{name}.csv", index=False)
    (out / "metrics.json").write_text(json.dumps(results, indent=2))
    print("test:", json.dumps(results["test"], indent=2))
    return results
