"""Experiment configuration.

Configs are YAML files mapped onto nested dataclasses. Any field can be
overridden from the CLI with dotted keys, e.g. ``--set drug.encoder=chemberta``.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    dataset: str = "davis"  # key in dti_fusion.data.datasets.DATASETS
    data_dir: str = "data"
    split: str = "random"  # random | cold_drug | cold_target | cold_both | scaffold
    split_seed: int = 0
    fractions: tuple[float, float, float] = (0.8, 0.1, 0.1)
    subsample: int | None = None  # keep N random pairs (smoke tests / quick runs)
    max_protein_len: int = 1000  # fusion sees at most this many residue tokens
    protein_len_policy: str = "pool"  # pool (adaptive avg over the full chain) | truncate
    max_smiles_tokens: int = 256


@dataclass
class ProteinConfig:
    esm_model: str = "facebook/esm2_t12_35M_UR50D"
    window: int = 1022  # ESM-2 positional limit; longer chains are embedded in overlapping windows
    window_overlap: int = 256
    tokens_per_batch: int = 8192


@dataclass
class GraphTransformerConfig:
    d_model: int = 256
    n_layers: int = 4
    n_heads: int = 8
    dropout: float = 0.1
    max_dist: int = 12  # shortest-path distances above this share one bias bucket


@dataclass
class ChemBERTaConfig:
    model: str = "DeepChem/ChemBERTa-77M-MLM"
    finetune: bool = False  # False: cached frozen token embeddings + trainable adapter
    adapter_layers: int = 1  # transformer layers on top of the (frozen) ChemBERTa tokens
    lr: float = 2.0e-5  # learning rate for ChemBERTa weights when finetune=True


@dataclass
class DrugConfig:
    encoder: str = "graph"  # graph | chemberta
    graph: GraphTransformerConfig = field(default_factory=GraphTransformerConfig)
    chemberta: ChemBERTaConfig = field(default_factory=ChemBERTaConfig)


@dataclass
class FusionConfig:
    mode: str = "cross"  # cross (bidirectional co-attention) | joint (concat + self-attention)
    d_model: int = 256
    n_layers: int = 2
    n_heads: int = 8
    ffn_mult: int = 4
    dropout: float = 0.1
    protein_self_attn: bool = False  # ESM tokens are already contextual; enable for an ablation


@dataclass
class TrainConfig:
    epochs: int = 60
    batch_size: int = 64
    lr: float = 3.0e-4
    weight_decay: float = 0.01
    warmup_frac: float = 0.05
    grad_clip: float = 1.0
    patience: int = 10  # early stopping on the selection metric
    select_metric: str = "rmse"  # validation metric used for checkpoint selection (lower is better)
    loss: str = "mse"  # mse | huber
    seed: int = 0
    device: str = "auto"  # auto | cuda | mps | cpu
    amp: str = "auto"  # auto (bf16 on CUDA if supported, off elsewhere) | off
    num_workers: int = 0
    out_dir: str = "runs"
    run_name: str | None = None


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    protein: ProteinConfig = field(default_factory=ProteinConfig)
    drug: DrugConfig = field(default_factory=DrugConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _build(cls, values: dict[str, Any]):
    kwargs = {}
    fields = {f.name: f for f in dataclasses.fields(cls)}
    for key, value in values.items():
        if key not in fields:
            raise KeyError(f"unknown config key '{key}' for {cls.__name__}")
        f = fields[key]
        default = f.default_factory() if f.default_factory is not dataclasses.MISSING else f.default
        if dataclasses.is_dataclass(default) and isinstance(value, dict):
            kwargs[key] = _build(type(default), value)
        elif isinstance(value, list):
            kwargs[key] = tuple(value)
        elif isinstance(default, float) and isinstance(value, (str, int)) and not isinstance(value, bool):
            kwargs[key] = float(value)  # PyYAML reads "1e-3" (no dot) as a string
        else:
            kwargs[key] = value
    return cls(**kwargs)


def _parse_scalar(text: str) -> Any:
    return yaml.safe_load(text)


def apply_overrides(raw: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"override must look like key.sub=value, got '{item}'")
        key, value = item.split("=", 1)
        node = raw
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = _parse_scalar(value)
    return raw


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> Config:
    raw: dict[str, Any] = {}
    if path is not None:
        raw = yaml.safe_load(Path(path).read_text()) or {}
    raw = apply_overrides(raw, overrides or [])
    return _build(Config, raw)
