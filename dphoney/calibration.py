"""DP-HONEY calibration + accounting (U4): split conformal + the canary relation.

Pure numpy. The quantile, coverage, and accounting functions operate on plain
score arrays, so they are testable in isolation without committing to a score
source. In the lab (U5) the nonconformity score is the distinguisher's bigram
log-likelihood of the most credential-shaped substring (``dphoney.distinguisher
.bigram_loglik``); a coverage number is only meaningful once that real score is
wired in — not over the synthetic placeholders these unit tests use.

Score convention: higher nonconformity = more canary-like (anomalous). The
conformal threshold is the ``1 - α`` upper quantile of benign scores, so at most
α of benign outputs exceed it (FPR ≤ α, coverage ≥ 1 − α). Conformal sets α
only; the miss rate β in the accounting relation is an independent detection
property the threshold does not set.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


def conformal_threshold(benign_scores: np.ndarray | list[float], alpha: float) -> float:
    """Split-conformal threshold: the finite-sample ``1 - α`` benign quantile.

    Uses the order statistic ``k = ceil((n + 1)(1 - α))`` (clamped to ``[1, n]``),
    which gives the standard marginal guarantee ``P(score_new > τ) ≤ α`` for a
    fresh benign point under exchangeability.
    """
    scores = np.sort(np.asarray(benign_scores, dtype=float))
    n = len(scores)
    if n == 0:
        raise ValueError("need at least one benign calibration score")
    k = int(np.ceil((n + 1) * (1.0 - alpha)))
    k = min(max(k, 1), n)
    return float(scores[k - 1])


def coverage(held_out_benign_scores: np.ndarray | list[float], threshold: float) -> float:
    """Empirical coverage: fraction of held-out benign scores at or below ``threshold``."""
    scores = np.asarray(held_out_benign_scores, dtype=float)
    return float(np.mean(scores <= threshold))


@dataclass(frozen=True)
class Calibration:
    alpha: float
    threshold: float
    coverage: float
    naive_threshold: float
    naive_coverage: float


def calibrate(
    calib_benign_scores: np.ndarray | list[float],
    eval_benign_scores: np.ndarray | list[float],
    alpha: float,
) -> Calibration:
    """Conformal threshold vs an untuned fixed (mean + 2σ) baseline.

    Conformal coverage lands near the ``1 - α`` target with no hand-tuning; the
    fixed-threshold baseline misses it because it does not adapt to ``α``.
    """
    calib = np.asarray(calib_benign_scores, dtype=float)
    tau = conformal_threshold(calib, alpha)
    naive = float(calib.mean() + 2.0 * calib.std())
    return Calibration(
        alpha=alpha,
        threshold=tau,
        coverage=coverage(eval_benign_scores, tau),
        naive_threshold=naive,
        naive_coverage=coverage(eval_benign_scores, naive),
    )


def detection_probability(k: int, m: int, beta: float) -> float:
    """Canary-accounting relation ``Pr(detect) = k/(m+k)·(1−β)``.

    ``k`` planted canaries, ``m`` real visible credentials, ``β`` detector miss
    rate. Probability that an attacker selecting one credential uniformly touches
    a canary and it is detected.
    """
    if k <= 0:
        return 0.0
    return (k / (m + k)) * (1.0 - beta)


def save_calibration(calib: Calibration, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(calib), indent=2))
    return path


def load_calibration(path: str | Path) -> Calibration:
    data = json.loads(Path(path).read_text())
    return Calibration(**data)
