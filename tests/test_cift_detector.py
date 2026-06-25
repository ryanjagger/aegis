from __future__ import annotations

import pytest

pytest.importorskip("sklearn")  # detector metrics need the opt-in cift group
np = pytest.importorskip("numpy")

from cift.detector import (  # noqa: E402
    evaluate_metrics,
    fit_baseline,
    fpr_at_tpr,
    load_baseline,
    load_operating_point,
    operating_point_from_benign_scores,
    per_layer_scores,
    save_baseline,
    save_operating_point,
    score,
)


def _synthetic(n: int, k: int, hidden: int, shift: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(loc=shift, scale=1.0, size=(n, k, hidden))


def test_attacks_score_higher_and_separate():
    # Covers R9: shifted attack features score higher and a held-out split separates.
    benign = _synthetic(200, 4, 32, shift=0.0, seed=1)
    attack = _synthetic(200, 4, 32, shift=1.2, seed=2)
    baseline = fit_baseline(benign[:160], ridge=1e-2)

    benign_scores = score(baseline, benign[160:])
    attack_scores = score(baseline, attack[160:])
    assert attack_scores.mean() > benign_scores.mean()

    scores = np.concatenate([benign_scores, attack_scores])
    labels = np.concatenate([np.zeros(len(benign_scores)), np.ones(len(attack_scores))])
    metrics = evaluate_metrics(scores, labels)
    assert metrics.auroc > 0.5


def test_metrics_are_bounded():
    # Covers R11: AUROC/F1/FPR all live in [0, 1]; FPR-at-TPR is reported.
    benign = _synthetic(120, 3, 16, shift=0.0, seed=3)
    attack = _synthetic(120, 3, 16, shift=1.5, seed=4)
    baseline = fit_baseline(benign[:90], ridge=1e-2)
    scores = np.concatenate([score(baseline, benign[90:]), score(baseline, attack[90:])])
    labels = np.concatenate([np.zeros(30), np.ones(30)])

    m = evaluate_metrics(scores, labels)
    for value in (m.auroc, m.f1, m.fpr):
        assert 0.0 <= value <= 1.0
    assert 0.0 <= fpr_at_tpr(scores, labels, 0.9) <= 1.0


def test_rank_deficient_stays_finite():
    # Edge: fewer samples than dims is the singular-covariance trap; diagonal+ridge
    # must keep scores finite rather than raising or returning NaN.
    benign = _synthetic(10, 2, 64, shift=0.0, seed=5)  # n=10 << hidden=64
    with pytest.warns(UserWarning):
        baseline = fit_baseline(benign, ridge=1e-2)
    s = score(baseline, _synthetic(5, 2, 64, shift=0.5, seed=6))
    assert np.all(np.isfinite(s))


def test_identical_distributions_have_no_signal():
    # Edge: when benign and "attack" are drawn identically, AUROC sits near 0.5.
    benign = _synthetic(300, 3, 24, shift=0.0, seed=7)
    other = _synthetic(150, 3, 24, shift=0.0, seed=8)
    baseline = fit_baseline(benign[:200], ridge=1e-2)
    scores = np.concatenate([score(baseline, benign[200:]), score(baseline, other)])
    labels = np.concatenate([np.zeros(100), np.ones(150)])
    assert abs(evaluate_metrics(scores, labels).auroc - 0.5) < 0.12


def test_empty_baseline_raises():
    # Error: an empty benign baseline is a usage error, not silent NaN scores.
    with pytest.raises(ValueError):
        fit_baseline(np.zeros((0, 3, 8)))


def test_per_layer_shape_and_single_prompt():
    baseline = fit_baseline(_synthetic(50, 4, 12, 0.0, 9), ridge=1e-2)
    assert per_layer_scores(baseline, _synthetic(7, 4, 12, 0.3, 10)).shape == (7, 4)
    # a single [K, hidden] prompt is accepted and yields one row
    assert per_layer_scores(baseline, _synthetic(1, 4, 12, 0.3, 11)[0]).shape == (1, 4)


def test_baseline_round_trips(tmp_path):
    baseline = fit_baseline(_synthetic(80, 3, 16, 0.0, 12), ridge=2e-2, fingerprint={"torch": "x"})
    path = save_baseline(baseline, tmp_path / "baseline.npz")
    loaded = load_baseline(path)
    assert np.allclose(loaded.mean, baseline.mean)
    assert loaded.ridge == baseline.ridge
    assert loaded.fingerprint == {"torch": "x"}


def test_operating_point_threshold_matches_target_fpr():
    # The benign-quantile threshold should leave ~target_fpr of benign above it.
    rng = np.random.default_rng(13)
    benign_scores = rng.normal(10.0, 2.0, size=2000)
    op = operating_point_from_benign_scores(benign_scores, target_fpr=0.05)
    empirical_fpr = float((benign_scores >= op.threshold).mean())
    assert abs(empirical_fpr - 0.05) < 0.02
    assert op.threshold > op.benign_mean  # an upper-tail cut sits above the mean


def test_operating_point_round_trips(tmp_path):
    op = operating_point_from_benign_scores(np.linspace(0.0, 100.0, 200), target_fpr=0.1)
    loaded = load_operating_point(save_operating_point(op, tmp_path / "threshold.json"))
    assert loaded.threshold == op.threshold
    assert loaded.benign_mean == op.benign_mean
    assert loaded.target_fpr == op.target_fpr


def test_operating_point_rejects_empty():
    with pytest.raises(ValueError):
        operating_point_from_benign_scores(np.array([]))
