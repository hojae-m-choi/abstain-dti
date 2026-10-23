"""Regression + derived-binary metrics for affinity prediction."""

from __future__ import annotations

import numpy as np
from scipy import stats
from sklearn.metrics import average_precision_score, roc_auc_score


def concordance_index(y_true: np.ndarray, y_pred: np.ndarray, chunk: int = 2048) -> float:
    """Fraction of comparable pairs (y_i > y_j) ranked correctly; prediction ties count 0.5.

    Exact O(n^2), chunked so 10^5 test pairs stay within memory.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    concordant, total = 0.0, 0.0
    for start in range(0, len(y_true), chunk):
        yt, yp = y_true[start : start + chunk, None], y_pred[start : start + chunk, None]
        comparable = yt > y_true[None, :]
        diff = yp - y_pred[None, :]
        concordant += np.sum(comparable & (diff > 0)) + 0.5 * np.sum(comparable & (diff == 0))
        total += np.sum(comparable)
    return float(concordant / total) if total else float("nan")


def regression_metrics(y_true, y_pred, binary_threshold: float | None = None) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    err = y_pred - y_true
    out = {
        "n": int(len(y_true)),
        "mse": float(np.mean(err**2)),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mae": float(np.mean(np.abs(err))),
        "pearson": float(stats.pearsonr(y_true, y_pred)[0]) if np.std(y_pred) > 0 else float("nan"),
        "spearman": float(stats.spearmanr(y_true, y_pred)[0]) if np.std(y_pred) > 0 else float("nan"),
        "ci": concordance_index(y_true, y_pred),
        "r2": float(1 - np.sum(err**2) / np.sum((y_true - y_true.mean()) ** 2)),
    }
    if binary_threshold is not None:
        labels = (y_true >= binary_threshold).astype(int)
        out["pos_rate"] = float(labels.mean())
        if 0 < labels.sum() < len(labels):
            out["auroc"] = float(roc_auc_score(labels, y_pred))
            out["auprc"] = float(average_precision_score(labels, y_pred))
    return out
