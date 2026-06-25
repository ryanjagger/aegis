"""Pure presentation helpers for the CIFT live monitor (no torch / streamlit).

Kept import-light and side-effect-free so the demo's display logic — gauge
scaling, verdict thresholding, leak-position lookup, layer labelling — is
unit-tested without loading a model. ``cift/live_monitor.py`` is the Streamlit
shell that wires these to the real extraction + scanner.
"""

from __future__ import annotations

import math


def gauge_fraction(score: float, threshold: float, spread: float) -> float:
    """Map a Mahalanobis score to a 0..1 gauge position.

    The threshold sits at 0.5; scores spread away from it via ``tanh`` scaled by
    the benign score spread, so a benign-typical score reads low and a clear
    anomaly saturates high without the bar pinning on the first outlier.
    """

    spread = spread if spread > 0 else 1.0
    x = (score - threshold) / (3.0 * spread)
    return max(0.0, min(1.0, 0.5 + 0.5 * math.tanh(x)))


def verdict_is_attack(score: float, threshold: float) -> bool:
    """True when the pre-output score is at or above the operating threshold."""

    return score >= threshold


def leak_char_index(text: str, secret: str | None) -> int | None:
    """Character index where ``secret`` first appears in ``text``, else None."""

    if not secret:
        return None
    idx = text.find(secret)
    return idx if idx >= 0 else None


def layer_labels(num_layers: int, k: int) -> list[str]:
    """Human labels ``L{n}`` for the last ``k`` monitored transformer layers."""

    start = num_layers - k + 1
    return [f"L{start + i}" for i in range(k)]
