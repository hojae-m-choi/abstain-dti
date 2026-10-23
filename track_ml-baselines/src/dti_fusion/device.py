"""Device selection that works on Apple Silicon (MPS), CUDA, and CPU."""

from __future__ import annotations

import contextlib
import os
import random

import numpy as np

# Let ops that MPS does not implement fall back to CPU instead of raising.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
# Cap MPS allocations at the device's recommended working set (default allows 1.7x),
# so an oversized run fails with an OOM error instead of swapping unified memory
# until the whole machine freezes. Raise it explicitly if you know you have headroom.
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "1.0")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.8")  # must stay <= high ratio

import torch  # noqa: E402


def resolve_device(preference: str = "auto") -> torch.device:
    if preference != "auto":
        return torch.device(preference)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def autocast(device: torch.device, amp: str = "auto"):
    """bf16 autocast on CUDA when supported; full precision on MPS/CPU.

    MPS fp16 autocast exists but is not reliable enough for training here,
    and bf16 needs no GradScaler, which keeps the loop identical across devices.
    """
    if amp != "off" and device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def inference_dtype(device: torch.device) -> torch.dtype:
    """dtype for frozen feature extractors (ESM-2, ChemBERTa)."""
    return torch.float16 if device.type == "cuda" else torch.float32


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # also seeds CUDA and MPS generators


def configure_backends() -> None:
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def describe() -> str:
    lines = [f"torch {torch.__version__}"]
    lines.append(f"cuda available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        lines.append(f"cuda device: {torch.cuda.get_device_name(0)}")
    lines.append(f"mps available: {torch.backends.mps.is_available()}")
    lines.append(f"selected: {resolve_device()}")
    return "\n".join(lines)
