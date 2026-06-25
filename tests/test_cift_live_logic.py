from __future__ import annotations

from cift.live_logic import (
    gauge_fraction,
    layer_labels,
    leak_char_index,
    verdict_is_attack,
)


def test_gauge_fraction_centers_on_threshold_and_is_bounded():
    # At the threshold the gauge reads the midpoint; it stays within [0, 1].
    assert gauge_fraction(10.0, 10.0, 2.0) == 0.5
    assert 0.0 <= gauge_fraction(-1e9, 10.0, 2.0) <= 0.5
    assert 0.5 <= gauge_fraction(1e9, 10.0, 2.0) <= 1.0


def test_gauge_fraction_monotonic_in_score():
    low = gauge_fraction(8.0, 10.0, 2.0)
    mid = gauge_fraction(10.0, 10.0, 2.0)
    high = gauge_fraction(14.0, 10.0, 2.0)
    assert low < mid < high


def test_gauge_fraction_handles_zero_spread():
    # A degenerate benign spread must not divide-by-zero.
    assert gauge_fraction(11.0, 10.0, 0.0) > 0.5


def test_verdict_is_attack_at_or_above_threshold():
    assert verdict_is_attack(10.0, 10.0) is True
    assert verdict_is_attack(10.1, 10.0) is True
    assert verdict_is_attack(9.9, 10.0) is False


def test_leak_char_index_finds_secret():
    assert leak_char_index("here is ghp_secret in the reply", "ghp_secret") == 8
    assert leak_char_index("no secret here", "ghp_secret") is None
    assert leak_char_index("anything", None) is None
    assert leak_char_index("anything", "") is None


def test_layer_labels_name_the_last_k_layers():
    # 28-layer model, K=7 -> the monitored window is L22..L28.
    assert layer_labels(28, 7) == ["L22", "L23", "L24", "L25", "L26", "L27", "L28"]
    assert layer_labels(24, 1) == ["L24"]
