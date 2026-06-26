"""Shared artifact paths for the DP-HONEY lab.

Light module — stdlib + app.config only, no numpy/sklearn/matplotlib — so the
base-env dashboard (U7) can locate lab outputs without pulling the opt-in stack.
"""

from __future__ import annotations

from pathlib import Path

from app.config import get_settings

SEPARABILITY_PNG = "separability.png"
COVERAGE_PNG = "coverage.png"
INTERPRETATION_MD = "interpretation.md"
BIGRAM_MODEL_NPZ = "bigram_model.npz"
CALIBRATION_JSON = "calibration.json"


def artifacts_dir() -> Path:
    """Where the lab writes figures, the model, calibration, and interpretation."""
    return Path(get_settings().data_dir) / "dphoney" / "artifacts"
