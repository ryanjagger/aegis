from __future__ import annotations

from typing import Any

from app.scanners.transforms import transformed_texts
from app.schemas.events import DetectorHit


def _attr_or_key(row: Any, name: str) -> Any:
    if isinstance(row, dict):
        return row.get(name)
    return getattr(row, name)


class CanaryScanner:
    def __init__(self, canaries: list[Any]) -> None:
        self.canaries = canaries

    @property
    def registered_values(self) -> list[str]:
        return [str(_attr_or_key(canary, "value")) for canary in self.canaries]

    def scan_text(self, text: str, surface: str) -> list[DetectorHit]:
        if not text or not self.canaries:
            return []

        hits: list[DetectorHit] = []
        seen: set[tuple[str, str, str]] = set()
        for detector, candidate in transformed_texts(text):
            for canary in self.canaries:
                canary_id = str(_attr_or_key(canary, "id"))
                value = str(_attr_or_key(canary, "value"))
                match_value = value
                if detector == "case_normalized_canary_match":
                    found = value.lower() in candidate
                elif detector == "whitespace_stripped_canary_match":
                    found = value in candidate
                else:
                    found = value in candidate

                if not found:
                    continue
                key = (detector, surface, canary_id)
                if key in seen:
                    continue
                seen.add(key)
                hits.append(
                    DetectorHit(
                        detector=detector,
                        surface=surface,
                        severity="critical",
                        matched_canary_id=canary_id,
                        evidence_preview=f"{detector} detected registered canary {canary_id}",
                        policy_recommendation="BLOCK",
                        matched_value=match_value,
                    )
                )
        return hits
