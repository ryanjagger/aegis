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


def footer_message(
    *, cift_flagged: bool, scanner_fired: bool, scannable: bool, leak_token: int | None
) -> str:
    """Honest bottom-line for a run, keyed on whether CIFT was actually *correct*.

    The naive "CIFT decided at token 0, scanner at token N" framing claims a
    pre-output advantage even when CIFT gave a false negative — which betrays the
    lab's no-overclaiming theme. This reports all four outcomes truthfully,
    including the case where only the post-output scanner caught a real leak.
    """

    where = f"token {leak_token}" if leak_token else "in the output"
    if not scannable:
        state = "flagged it as credential-seeking" if cift_flagged else "read it as benign"
        return (
            f"No secret was registered to scan for, so only CIFT ran — it {state} at token 0, "
            "before any output existed."
        )
    if cift_flagged and scanner_fired:
        return (
            f"CIFT flagged this at token 0 — the text scanner only confirmed the leak at {where}. "
            "That window is the pre-output advantage."
        )
    if cift_flagged and not scanner_fired:
        return (
            "CIFT flagged the intent pre-output, but the model did not emit the secret in this "
            "sample — the activation probe reacts to intent, not only to a realized leak."
        )
    if scanner_fired:  # not cift_flagged: a genuine miss
        return (
            f"CIFT missed this — only the text scanner caught the leak ({where}). The subtle "
            "signal fell below the operating threshold; at this model scale CIFT trades recall "
            "for a usable false-positive rate."
        )
    return (
        "CIFT and the text scanner both read this as clean — either genuinely benign, or a subtle "
        "case that slipped past both."
    )
