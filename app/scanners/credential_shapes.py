from __future__ import annotations

import re

from app.schemas.events import DetectorHit

CREDENTIAL_PATTERNS = {
    "github_pat_shape": re.compile(r"\bghp_[A-Za-z0-9_]{20,}\b"),
    "stripe_key_shape": re.compile(r"\bsk_live_[A-Za-z0-9_]{16,}\b"),
    "aws_access_key_shape": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "postgres_url_shape": re.compile(r"postgres://[^\s'\"<>]+"),
    "jwt_like_shape": re.compile(
        r"\b[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b"
    ),
    "support_token_shape": re.compile(r"\bsupport_live_[A-Za-z0-9_]{16,}\b"),
}


class CredentialShapeDetector:
    def __init__(self, registered_values: list[str] | None = None) -> None:
        self.registered_values = set(registered_values or [])

    def scan_text(self, text: str, surface: str) -> list[DetectorHit]:
        hits: list[DetectorHit] = []
        seen: set[tuple[str, str]] = set()
        for detector, pattern in CREDENTIAL_PATTERNS.items():
            for match in pattern.finditer(text):
                value = match.group(0)
                if value in self.registered_values:
                    continue
                key = (detector, value)
                if key in seen:
                    continue
                seen.add(key)
                hits.append(
                    DetectorHit(
                        detector=detector,
                        surface=surface,
                        severity="medium",
                        matched_canary_id=None,
                        evidence_preview=f"Credential-shaped value detected by {detector}",
                        policy_recommendation="SANITIZE",
                        matched_value=value,
                    )
                )
        return hits
