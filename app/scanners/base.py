from __future__ import annotations

from typing import Protocol

from app.schemas.events import DetectorHit


class TextScanner(Protocol):
    def scan_text(self, text: str, surface: str) -> list[DetectorHit]:
        ...
