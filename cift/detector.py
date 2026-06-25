"""Unsupervised Mahalanobis baseline detector and metrics (U4).

The core CIFT detector. It is the unsupervised baseline that ships before any
learned probe: fit a benign mean/std per (layer, hidden-dim), then score each
prompt by its per-layer diagonal Mahalanobis distance (ridge-regularized) summed
across the last-K layers. Higher score = more credential-access-like.

All maths run in NumPy float64 on CPU (MPS has no float64, and covariance work
wants the precision). Features are ``[N, K, hidden]`` arrays produced by
``cift.extraction``; this module never touches a model, so it is unit-testable on
synthetic arrays.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, roc_curve

_EPS = 1e-8


@dataclass(frozen=True)
class MahalanobisBaseline:
    """Per-(layer, dim) benign statistics for the diagonal Mahalanobis score."""

    mean: np.ndarray  # [K, hidden] benign feature mean
    std: np.ndarray  # [K, hidden] benign feature std (for standardization)
    var: np.ndarray  # [K, hidden] standardized-feature variance + ridge
    ridge: float
    fingerprint: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Metrics:
    auroc: float
    f1: float
    fpr: float  # false-positive rate at the chosen operating point
    threshold: float

    def as_dict(self) -> dict:
        return {"auroc": self.auroc, "f1": self.f1, "fpr": self.fpr, "threshold": self.threshold}


def fit_baseline(
    benign_feats: np.ndarray, *, ridge: float = 1e-2, fingerprint: dict | None = None
) -> MahalanobisBaseline:
    """Fit benign statistics from ``[N, K, hidden]`` benign features.

    Full covariance on hidden-size dims from a few hundred samples is
    rank-deficient and non-invertible, so we use diagonal covariance plus a ridge
    term — the standard fix. Features are standardized (z-scored) on the benign
    set first, so the per-dim variance is ~1 before the ridge floor.
    """

    feats = np.asarray(benign_feats, dtype=np.float64)
    if feats.ndim != 3 or feats.shape[0] == 0:
        raise ValueError(
            "benign_feats must be a non-empty [N, K, hidden] array; "
            f"got shape {getattr(feats, 'shape', None)}"
        )
    n = feats.shape[0]
    mean = feats.mean(axis=0)
    std = feats.std(axis=0) + _EPS
    z = (feats - mean) / std
    var = z.var(axis=0) + ridge

    if n < feats.shape[2]:
        warnings.warn(
            f"benign n={n} is below hidden_size={feats.shape[2]}; per-dim statistics are "
            "noisy. Diagonal + ridge keeps scores finite, but consider a larger benign set.",
            stacklevel=2,
        )
    return MahalanobisBaseline(
        mean=mean, std=std, var=var, ridge=ridge, fingerprint=fingerprint or {}
    )


def per_layer_scores(baseline: MahalanobisBaseline, feats: np.ndarray) -> np.ndarray:
    """Per-layer diagonal Mahalanobis^2 distance: ``[N, K]``."""

    feats = np.asarray(feats, dtype=np.float64)
    if feats.ndim == 2:  # a single prompt [K, hidden] -> [1, K, hidden]
        feats = feats[None, :, :]
    z = (feats - baseline.mean) / baseline.std
    return ((z * z) / baseline.var).sum(axis=2)  # [N, K]


def score(baseline: MahalanobisBaseline, feats: np.ndarray) -> np.ndarray:
    """Aggregate anomaly score per prompt: sum of per-layer distances, ``[N]``."""

    return per_layer_scores(baseline, feats).sum(axis=1)


def evaluate_metrics(scores: np.ndarray, labels: np.ndarray) -> Metrics:
    """AUROC, plus F1/FPR at the Youden-J operating point (1 = attack)."""

    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if len(np.unique(labels)) < 2:
        raise ValueError("labels must contain both classes (0 benign, 1 attack)")

    auroc = float(roc_auc_score(labels, scores))
    fpr_grid, tpr_grid, thresholds = roc_curve(labels, scores)
    youden = tpr_grid - fpr_grid
    best = int(np.argmax(youden))
    threshold = float(thresholds[best])
    preds = (scores >= threshold).astype(np.int64)
    f1 = float(f1_score(labels, preds, zero_division=0))
    fpr = float(fpr_grid[best])
    return Metrics(auroc=auroc, f1=f1, fpr=fpr, threshold=threshold)


def fpr_at_tpr(scores: np.ndarray, labels: np.ndarray, target_tpr: float = 0.95) -> float:
    """Lowest FPR achieving at least ``target_tpr`` true-positive rate."""

    fpr_grid, tpr_grid, _ = roc_curve(np.asarray(labels), np.asarray(scores, dtype=np.float64))
    eligible = fpr_grid[tpr_grid >= target_tpr]
    return float(eligible.min()) if eligible.size else 1.0


def save_baseline(baseline: MahalanobisBaseline, path: str | Path) -> Path:
    """Persist the baseline (and its fingerprint) as an ``.npz`` next to a sidecar."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        mean=baseline.mean,
        std=baseline.std,
        var=baseline.var,
        ridge=np.array(baseline.ridge),
        fingerprint=np.array(json.dumps(baseline.fingerprint)),
    )
    return path


def load_baseline(path: str | Path) -> MahalanobisBaseline:
    data = np.load(Path(path), allow_pickle=False)
    return MahalanobisBaseline(
        mean=data["mean"],
        std=data["std"],
        var=data["var"],
        ridge=float(data["ridge"]),
        fingerprint=json.loads(str(data["fingerprint"])),
    )
