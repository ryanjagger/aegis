"""U5: gated interpretation, figures, and the end-to-end lab.

The lab is a group module (imports sklearn + matplotlib), so the whole file is
gated; the gating logic of build_interpretation is the test-first core.
"""

from __future__ import annotations

import pytest

pytest.importorskip("sklearn")
pytest.importorskip("matplotlib")

from dphoney import figures  # noqa: E402
from dphoney.distinguisher import ControlResult  # noqa: E402
from dphoney.run import build_interpretation, run_lab  # noqa: E402

PASS = ControlResult("ctrl", 0.99, True)
FAIL = ControlResult("ctrl", 0.50, False)


def _interp(battery, left_half, **kw):
    defaults = dict(
        template_auroc=1.0, dp_auroc=0.5, epsilon=50.0, coverage=0.99, target_coverage=0.99
    )
    defaults.update(kw)
    return build_interpretation(battery, left_half, **defaults)


def test_interpretation_gates_on_failed_battery():
    text = _interp(FAIL, PASS)
    assert "not interpretable" in text.lower()
    assert "gap" not in text.lower()  # the contrast is not reported


def test_interpretation_flags_structureless_reference():
    text = _interp(PASS, FAIL)
    assert "not a real null" in text.lower()
    assert "structureless" in text.lower()


def test_interpretation_reports_contrast_when_controls_pass():
    text = _interp(PASS, PASS, template_auroc=0.98, dp_auroc=0.55)
    assert "0.98" in text and "0.55" in text
    assert "ε = 50" in text
    assert "coverage 0.990" in text
    assert "indistinguishability" in text  # the honest ε-DP scope caveat


def test_figures_write_nonempty_png(tmp_path):
    sep = figures.plot_separability_contrast(
        {"github_pat": (1.0, 0.5), "aws_access_key": (0.99, 0.6)}, tmp_path / "sep.png"
    )
    cov = figures.plot_coverage(0.95, 0.99, 0.99, tmp_path / "cov.png")
    assert sep.exists() and sep.stat().st_size > 0
    assert cov.exists() and cov.stat().st_size > 0


def test_run_lab_smoke_produces_contrast():
    result = run_lab(seed=0, n_per_format=160, alpha=0.05)
    assert result.battery.passed
    assert result.left_half.passed
    assert result.template_auroc > result.dp_auroc  # the headline contrast
    text = build_interpretation(
        result.battery,
        result.left_half,
        template_auroc=result.template_auroc,
        dp_auroc=result.dp_auroc,
        epsilon=result.epsilon,
        coverage=result.calibration.coverage,
        target_coverage=1 - result.calibration.alpha,
    )
    assert "interpretation" in text.lower()
