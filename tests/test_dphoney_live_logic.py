"""U7: pure walkthrough helpers (no streamlit). Runs on the base env."""

from __future__ import annotations

import pytest

from dphoney import live_logic
from dphoney.calibration import detection_probability


def test_accounting_matches_relation():  # R9
    for k, m, beta in [(5, 5, 0.1), (0, 3, 0.2), (10, 1, 0.0), (3, 3, 1.0)]:
        assert live_logic.accounting(k, m, beta) == pytest.approx(detection_probability(k, m, beta))


def test_nimbus_label_is_honest():  # R13
    label = live_logic.nimbus_label()
    assert "NIMBUS-lite" in label
    assert "InfoNCE" in label
    assert "NOT" in label  # explicitly distinguished from the paper's estimator


def test_prerun_messages_signal_no_run_yet():
    assert "No scenario run yet" in live_logic.prerun_message("text backstop")
    assert "No scenario run yet" in live_logic.prerun_message("NIMBUS")
    assert "Run the DP-HONEY lab first" in live_logic.prerun_message("DP-HONEY")


def test_separability_caption_degrades_gracefully():
    assert "Run the DP-HONEY lab first" in live_logic.separability_caption(False)
    assert "indistinguishable" in live_logic.separability_caption(True)


def test_station_order_is_the_pipeline():
    assert live_logic.station_order() == ["DP-HONEY", "CIFT", "text backstop", "NIMBUS"]


def test_module_stays_light():
    import pathlib

    src = pathlib.Path(live_logic.__file__).read_text()
    imports = "\n".join(
        line for line in src.splitlines() if line.strip().startswith(("import ", "from "))
    )
    assert "streamlit" not in imports
    assert "torch" not in imports and "sklearn" not in imports
