"""U4: split-conformal calibration + canary accounting. Pure numpy."""

from __future__ import annotations

import numpy as np
import pytest

from dphoney.calibration import (
    calibrate,
    conformal_threshold,
    coverage,
    detection_probability,
    load_calibration,
    save_calibration,
)


def test_conformal_coverage_lands_near_target():  # AE2
    rng = np.random.default_rng(0)
    calib = rng.normal(size=2000)
    held_out = rng.normal(size=2000)
    alpha = 0.01
    calibration = calibrate(calib, held_out, alpha)
    assert abs(calibration.coverage - (1 - alpha)) < 0.02
    # Conformal lands closer to the target than the untuned fixed threshold.
    target = 1 - alpha
    assert abs(calibration.coverage - target) <= abs(calibration.naive_coverage - target)


def test_quantile_matches_closed_form():
    scores = [float(i) for i in range(1, 11)]  # 1..10
    alpha = 0.2
    n = len(scores)
    k = int(np.ceil((n + 1) * (1 - alpha)))  # ceil(11*0.8)=9
    assert conformal_threshold(scores, alpha) == sorted(scores)[k - 1]


def test_quantile_clamps_at_extremes():
    scores = [5.0]
    # n=1: k clamps to 1; threshold is the single score regardless of alpha.
    assert conformal_threshold(scores, 0.01) == 5.0
    assert conformal_threshold(scores, 0.99) == 5.0


def test_coverage_is_fraction_below_threshold():
    assert coverage([0.0, 1.0, 2.0, 3.0], threshold=1.0) == 0.5


def test_detection_probability_edge_cases():  # R9
    assert detection_probability(0, 5, 0.1) == 0.0  # no canaries planted
    assert detection_probability(2, 2, 0.0) == pytest.approx(0.5)  # β=0 -> k/(m+k)
    assert detection_probability(3, 1, 1.0) == 0.0  # β=1 -> never detected
    # Monotonic increasing in k.
    probs = [detection_probability(k, 5, 0.1) for k in range(0, 6)]
    assert all(b >= a for a, b in zip(probs, probs[1:], strict=False))


def test_calibration_round_trips(tmp_path):
    rng = np.random.default_rng(1)
    calibration = calibrate(rng.normal(size=500), rng.normal(size=500), 0.05)
    path = tmp_path / "calibration.json"
    save_calibration(calibration, path)
    loaded = load_calibration(path)
    assert loaded == calibration
