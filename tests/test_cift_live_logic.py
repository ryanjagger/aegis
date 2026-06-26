from __future__ import annotations

from cift.live_logic import (
    footer_message,
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


def test_footer_claims_advantage_only_when_cift_was_right():
    msg = footer_message(cift_flagged=True, scanner_fired=True, scannable=True, leak_token=63)
    assert "pre-output advantage" in msg
    assert "token 63" in msg


def test_footer_calls_a_miss_a_miss():
    # The key honesty case: CIFT said benign, the scanner caught a real leak.
    msg = footer_message(cift_flagged=False, scanner_fired=True, scannable=True, leak_token=63)
    assert "missed" in msg.lower()
    assert "advantage" not in msg.lower()
    assert "token 63" in msg


def test_footer_intent_without_realized_leak():
    msg = footer_message(cift_flagged=True, scanner_fired=False, scannable=True, leak_token=None)
    assert "did not emit the secret" in msg
    assert "advantage" not in msg.lower()


def test_footer_both_clean():
    msg = footer_message(cift_flagged=False, scanner_fired=False, scannable=True, leak_token=None)
    assert "both read this as clean" in msg


def test_footer_no_secret_registered_reports_cift_only():
    flagged = footer_message(
        cift_flagged=True, scanner_fired=False, scannable=False, leak_token=None
    )
    assert "only CIFT ran" in flagged
    assert "credential-seeking" in flagged
    benign = footer_message(
        cift_flagged=False, scanner_fired=False, scannable=False, leak_token=None
    )
    assert "read it as benign" in benign
