"""On-disk format for variable-length token embeddings.

Kept free of torch on purpose: readers such as the FLAML arm of
experiments/exp2-comparing_models1 need the embeddings but must not load torch,
because torch and LightGBM each pull in their own OpenMP runtime and LightGBM
then segfaults when it starts its threads (macOS arm64).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _slug(model_name: str) -> str:
    return model_name.replace("/", "__")


def protein_store_path(data_dir: str | Path, dataset: str, model_name: str) -> Path:
    return Path(data_dir) / "features" / dataset / f"esm-{_slug(model_name)}"


def chemberta_store_path(data_dir: str | Path, dataset: str, model_name: str, max_tokens: int) -> Path:
    return Path(data_dir) / "features" / dataset / f"chemberta-{_slug(model_name)}-T{max_tokens}"


class TokenStore:
    """Many variable-length [L, D] embeddings in one float16 memmap plus an offset index."""

    def __init__(self, root: Path):
        self.root = Path(root)
        meta = json.loads((self.root / "index.json").read_text())
        self.index: dict[str, tuple[int, int]] = {k: tuple(v) for k, v in meta["index"].items()}
        self.dim = meta["dim"]
        total = meta["total"]
        self.matrix = np.memmap(self.root / "emb.f16", dtype=np.float16, mode="r", shape=(total, self.dim))

    @staticmethod
    def exists(root: Path) -> bool:
        return (Path(root) / "index.json").exists()

    def __contains__(self, key: str) -> bool:
        return key in self.index

    def get(self, key: str) -> np.ndarray:
        offset, length = self.index[key]
        return self.matrix[offset : offset + length]

    @staticmethod
    def write(root: Path, items: dict[str, np.ndarray], extra: dict | None = None) -> "TokenStore":
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        dim = next(iter(items.values())).shape[1]
        total = sum(v.shape[0] for v in items.values())
        matrix = np.memmap(root / "emb.f16", dtype=np.float16, mode="w+", shape=(total, dim))
        index, offset = {}, 0
        for key, arr in items.items():
            matrix[offset : offset + len(arr)] = arr.astype(np.float16)
            index[key] = (offset, len(arr))
            offset += len(arr)
        matrix.flush()
        meta = {"dim": dim, "total": total, "index": index, **(extra or {})}
        (root / "index.json").write_text(json.dumps(meta))
        return TokenStore(root)
