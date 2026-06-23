from __future__ import annotations

from app.schemas.events import DetectorHit


def score_hits(hits: list[DetectorHit]) -> float:
    score = 0.0
    for hit in hits:
        if hit.matched_canary_id and hit.detector.startswith("exact"):
            score += 10.0
        elif hit.matched_canary_id:
            score += 8.0
        elif hit.severity == "medium":
            score += 3.0
    return score
