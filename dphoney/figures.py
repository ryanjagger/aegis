"""DP-HONEY figures (U5): the separability contrast and conformal coverage.

Matplotlib with the Agg backend (headless), mirroring cift/figures.py. Needs the
opt-in ``dphoney`` group.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def plot_separability_contrast(
    per_format: dict[str, tuple[float, float]], path: str | Path
) -> Path:
    """Grouped bars: template vs DP discriminator AUROC per format (chance = 0.5)."""
    names = list(per_format)
    template = [per_format[n][0] for n in names]
    dp = [per_format[n][1] for n in names]
    x = np.arange(len(names))
    width = 0.38

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.bar(x - width / 2, template, width, label="template canaries", color="#e45756")
    ax.bar(x + width / 2, dp, width, label="DP canaries", color="#4c78a8")
    ax.axhline(0.5, ls="--", color="gray", lw=1, label="chance (indistinguishable)")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel("distinguisher AUROC\n(lower = harder to filter)")
    ax.set_ylim(0, 1.05)
    ax.set_title("DP-HONEY: can an attacker tell the canaries from real-format credentials?")
    ax.legend(loc="lower right")
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_coverage(
    naive_coverage: float, conformal_coverage: float, target: float, path: str | Path
) -> Path:
    """Bars: untuned fixed threshold vs conformal, against the coverage target."""
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.bar(
        ["untuned\n(mean+2σ)", "conformal"],
        [naive_coverage, conformal_coverage],
        color=["#bbbbbb", "#4c78a8"],
    )
    ax.axhline(target, ls="--", color="#e45756", lw=1.5, label=f"target {target:.2f}")
    ax.set_ylabel("coverage on held-out benign")
    ax.set_ylim(0, 1.05)
    ax.set_title("Conformal calibration hits the target without hand-tuning")
    ax.legend(loc="lower right")
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
