from __future__ import annotations


def zone_for_score(score_total: float, budget: float = 10.0) -> str:
    ratio = score_total / budget if budget else 0.0
    if ratio >= 1.0:
        return "BLOCK"
    if ratio >= 0.8:
        return "SANITIZE"
    if ratio >= 0.6:
        return "WARN"
    return "PASS"
