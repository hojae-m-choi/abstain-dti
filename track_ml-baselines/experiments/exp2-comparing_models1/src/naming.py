"""How a run's dataset size is written down, in one place.

Two independent knobs decide what a model saw: how many targets were kept
(``top_targets``) and how many pairs were sampled from inside them (``n``). Both go
into one tag, so grids run at different settings sit side by side instead of
overwriting each other's subsample or being averaged together by 06_summarize.py:

    t300n100000  ->  300 busiest targets, 100,000 pairs

The ESM-2 / ChemBERTa caches deliberately stay outside the tag: a ``TokenStore`` is a
key -> embedding map, so a larger ``top_targets`` just rebuilds it as a superset and
every smaller tag keeps reading the same rows.
"""

from __future__ import annotations

import re
from pathlib import Path

TAG_RE = re.compile(r"^t(\d+)n(\d+)$")


def tag(top_targets: int, n: int) -> str:
    return f"t{top_targets}n{n}"


def parse_tag(text: str) -> tuple[int, int]:
    m = TAG_RE.match(text)
    if not m:
        raise ValueError(f"not a size tag: {text!r} (expected e.g. t300n100000)")
    return int(m.group(1)), int(m.group(2))


def subsample_path(exp_dir: Path, size_tag: str) -> Path:
    return Path(exp_dir) / "data" / f"sub_{size_tag}.parquet"


def split_path(exp_dir: Path, size_tag: str, kind: str, seed: int) -> Path:
    return Path(exp_dir) / "data" / "splits" / f"{size_tag}_{kind}_s{seed}.json"


def run_name(size_tag: str, split: str, arm: str, seed: int) -> str:
    return f"{size_tag}_{split}_{arm}_s{seed}"


def parse_run_name(name: str, split_order: tuple[str, ...] | list[str]) -> dict:
    """Inverse of :func:`run_name`; split names contain underscores themselves."""
    size_tag, rest = name.split("_", 1)
    top_targets, n = parse_tag(size_tag)
    arm, seed = rest.rsplit("_s", 1)
    split = next((s for s in split_order if arm.startswith(s + "_")), None)
    if split:
        arm = arm[len(split) + 1 :]
    return {
        "tag": size_tag,
        "top_targets": top_targets,
        "n": n,
        "split": split,
        "arm": arm,
        "seed": int(seed),
    }
