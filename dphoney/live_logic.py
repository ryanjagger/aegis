"""Pure helpers for the AIS Walkthrough page (U7).

No streamlit, no sklearn/matplotlib — testable on the base env. The streamlit
page (dashboard/streamlit_app.py) renders these; the behaviour lives here so the
copy, ordering, empty states, and the accounting wiring are unit-tested without a
browser. Mirrors cift/live_logic.py's separation of pure logic from rendering.
"""

from __future__ import annotations

from dphoney.calibration import detection_probability

STATIONS: tuple[str, ...] = ("DP-HONEY", "CIFT", "text backstop", "NIMBUS")

NIMBUS_LITE_LABEL = (
    "NIMBUS-lite — a heuristic cumulative-leakage score, NOT the paper's InfoNCE "
    "mutual-information estimator. It demonstrates the accounting structure "
    "(warn / sanitize / block) with hand-weighted points, not learned bits."
)

CIFT_LAUNCH = "uv run --group cift streamlit run cift/live_monitor.py"

_PRERUN: dict[str, str] = {
    "DP-HONEY": (
        "Run the DP-HONEY lab first "
        "(uv run --group dphoney python -m dphoney.run) to populate the figure."
    ),
    "CIFT": "CIFT is the opt-in white-box station — launch it separately.",
    "text backstop": "No scenario run yet — inject a DP canary above to see the scanner verdict.",
    "NIMBUS": "No scenario run yet — the cumulative ledger is empty.",
}


def station_order() -> list[str]:
    return list(STATIONS)


def accounting(k: int, m: int, beta: float) -> float:
    """Pr(detect) = k/(m+k)·(1−β), delegating to the calibration relation."""
    return detection_probability(k, m, beta)


def prerun_message(station: str) -> str:
    return _PRERUN[station]


def separability_caption(figure_exists: bool) -> str:
    if figure_exists:
        return "Template canaries are filterable; DP canaries sit at chance — indistinguishable."
    return _PRERUN["DP-HONEY"]


def nimbus_label() -> str:
    return NIMBUS_LITE_LABEL


def cift_launch_command() -> str:
    return CIFT_LAUNCH
