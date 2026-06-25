"""Lab figures (U7): per-layer Mahalanobis deviation and the encoding contrast.

Uses a non-interactive matplotlib backend so figures render headless. Both
functions operate on plain arrays / the contrast rows, so they are testable on
synthetic data without a model.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from cift.evaluate import ContrastRow  # noqa: E402


def plot_per_layer_mahalanobis(
    benign_per_layer: np.ndarray, attack_per_layer: np.ndarray, path: str | Path
) -> Path:
    """Per-layer mean Mahalanobis deviation, benign vs attack: the figure that
    shows which late layers carry the credential-access signal."""

    benign = np.asarray(benign_per_layer, dtype=np.float64)
    attack = np.asarray(attack_per_layer, dtype=np.float64)
    layers = np.arange(benign.shape[1])

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(layers, benign.mean(axis=0), marker="o", label="benign")
    ax.plot(layers, attack.mean(axis=0), marker="o", label="credential-seeking")
    ax.fill_between(
        layers,
        attack.mean(axis=0) - attack.std(axis=0),
        attack.mean(axis=0) + attack.std(axis=0),
        alpha=0.15,
    )
    ax.set_xlabel("monitored layer (last K)")
    ax.set_ylabel("per-layer Mahalanobis distance")
    ax.set_title("CIFT per-layer deviation: benign vs credential-seeking")
    ax.legend()
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_encoding_robustness(rows: dict[str, ContrastRow], path: str | Path) -> Path:
    """Grouped bars of text-scanner F1 vs CIFT F1 per encoding — the contrast.

    A good result shows the text bar collapsing from verbatim to rot13 while the
    CIFT bar stays roughly flat.
    """

    encodings = list(rows.keys())
    text_f1 = [rows[e].text_f1 for e in encodings]
    cift_f1 = [rows[e].cift_f1 for e in encodings]
    x = np.arange(len(encodings))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - width / 2, text_f1, width, label="text scanner")
    ax.bar(x + width / 2, cift_f1, width, label="CIFT (activations)")
    ax.set_xticks(x)
    ax.set_xticklabels(encodings)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("detection F1")
    ax.set_title("Encoding robustness: text scanner vs CIFT")
    ax.legend()
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
