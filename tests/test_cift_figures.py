from __future__ import annotations

import pytest

pytest.importorskip("matplotlib")
pytest.importorskip("sklearn")
np = pytest.importorskip("numpy")

from cift.detector import Metrics  # noqa: E402
from cift.evaluate import ContrastRow, ControlResult  # noqa: E402
from cift.figures import plot_encoding_robustness, plot_per_layer_mahalanobis  # noqa: E402
from cift.run import build_interpretation  # noqa: E402


def test_per_layer_figure_written(tmp_path):
    # Covers R13: a per-layer figure renders for synthetic deviation data.
    benign = np.random.default_rng(0).normal(1.0, 0.2, size=(20, 7))
    attack = np.random.default_rng(1).normal(2.0, 0.3, size=(20, 7))
    path = plot_per_layer_mahalanobis(benign, attack, tmp_path / "per_layer.png")
    assert path.exists() and path.stat().st_size > 0


def test_encoding_figure_written(tmp_path):
    # Covers R13: the encoding-robustness contrast figure renders.
    rows = {
        "verbatim": ContrastRow("verbatim", text_f1=0.95, cift_f1=0.92, n=40),
        "rot13": ContrastRow("rot13", text_f1=0.05, cift_f1=0.90, n=40),
    }
    path = plot_encoding_robustness(rows, tmp_path / "encoding.png")
    assert path.exists() and path.stat().st_size > 0


def test_interpretation_refuses_null_when_control_failed():
    # Covers R14: a failed positive control blocks interpreting the main result.
    text = build_interpretation(
        ControlResult(auroc=0.52, passed=False), None, None, None, None
    )
    assert "not interpretable" in text.lower()
    assert "FAILED" in text


def test_interpretation_reports_result_when_control_passed():
    rows = {
        "verbatim": ContrastRow("verbatim", 0.9, 0.9, 40),
        "rot13": ContrastRow("rot13", 0.1, 0.88, 40),
    }
    text = build_interpretation(
        ControlResult(auroc=0.99, passed=True),
        Metrics(auroc=0.83, f1=0.78, fpr=0.04, threshold=12.0),
        rows,
        0.7,
        np.array([0.1, 0.2, 1.4, 0.3]),
    )
    assert "AUROC 0.83" in text
    assert "rot13: text 0.10, CIFT 0.88" in text
    assert "encoding success rate" in text.lower()
    assert "late layers carrying the signal" in text


def test_interpretation_does_not_overclaim_on_null_result():
    # A near-chance AUROC must NOT claim late-layer signal, even with a control pass.
    text = build_interpretation(
        ControlResult(auroc=1.0, passed=True),
        Metrics(auroc=0.27, f1=0.0, fpr=0.0, threshold=5.0),
        None,
        None,
        np.array([-40.0, -47.0, -10.0]),  # negative gaps = no separation
    )
    assert "no meaningful separation" in text.lower()
    assert "late layers carrying the signal" not in text
