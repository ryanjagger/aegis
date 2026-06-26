from __future__ import annotations

import pytest

pytest.importorskip("sklearn")
np = pytest.importorskip("numpy")

from app.canaries.generator import generate_canary  # noqa: E402
from app.scanners.canary_scanner import CanaryScanner  # noqa: E402
from cift.corpus import rot13  # noqa: E402
from cift.evaluate import (  # noqa: E402
    ContrastRecord,
    compute_contrast,
    positive_control_from_features,
    text_detected,
)


def _feats(n, k, hidden, shift, seed):
    return np.random.default_rng(seed).normal(shift, 1.0, size=(n, k, hidden))


def test_positive_control_passes_on_separable_case():
    # Covers R10: a clearly-separable case clears the control floor.
    benign = _feats(80, 4, 32, 0.0, 1)
    attack = _feats(40, 4, 32, 2.0, 2)
    result = positive_control_from_features(benign, attack)
    assert result.passed is True
    assert result.auroc >= 0.9
    assert result.verdict == "passed"


def test_positive_control_fails_on_broken_pipeline():
    # Covers R10: a broken extraction (no separation) is detected as control-FAILED,
    # so a null main result is not silently read as "CIFT doesn't work here".
    benign = _feats(80, 4, 32, 0.0, 3)
    attack = _feats(40, 4, 32, 0.0, 4)  # identical distribution = nothing to detect
    result = positive_control_from_features(benign, attack)
    assert result.passed is False
    assert result.verdict == "FAILED"


def test_contrast_text_collapses_while_cift_holds():
    # Covers R12 / AE1: text arm F1 drops sharply from verbatim to rot13 while the
    # CIFT arm stays roughly flat (the paper's centerpiece result, on synthetic
    # arm votes).
    records: list[ContrastRecord] = []
    for _ in range(20):  # attacks
        records.append(ContrastRecord("verbatim", 1, text_hit=True, cift_hit=True))
        records.append(ContrastRecord("rot13", 1, text_hit=False, cift_hit=True))
    for _ in range(20):  # benign negatives
        records.append(ContrastRecord("verbatim", 0, text_hit=False, cift_hit=False))
        records.append(ContrastRecord("rot13", 0, text_hit=False, cift_hit=False))

    rows = compute_contrast(records)
    assert rows["verbatim"].text_f1 > rows["rot13"].text_f1  # text collapses
    assert rows["rot13"].text_f1 == 0.0
    assert rows["rot13"].cift_f1 >= rows["verbatim"].cift_f1 - 1e-9  # CIFT holds flat


def test_contrast_marks_encoding_unevaluable_with_no_attacks():
    # PR review: if every encoded attack is dropped (e.g. 0% rot13 success), the
    # rot13 subset is all-benign and its F1 is degenerate — mark it unevaluable
    # rather than reporting a misleading 0.0 detector failure.
    records: list[ContrastRecord] = []
    for _ in range(10):
        records.append(ContrastRecord("verbatim", 1, text_hit=True, cift_hit=True))
        records.append(ContrastRecord("verbatim", 0, text_hit=False, cift_hit=False))
        records.append(ContrastRecord("rot13", 0, text_hit=False, cift_hit=False))  # benign only

    rows = compute_contrast(records)
    assert rows["verbatim"].evaluable is True
    assert rows["rot13"].evaluable is False  # no attack positives survived


def test_text_detected_uses_real_scanner():
    canary = generate_canary("github_pat", "lab")
    scanner = CanaryScanner([canary])
    assert text_detected(scanner, f"the key is {canary.value} ok")
    assert not text_detected(scanner, "nothing secret here")
    # the rot13'd secret is invisible to the text scanner (held-out encoding)
    assert not text_detected(scanner, f"encoded: {rot13(canary.value)}")
