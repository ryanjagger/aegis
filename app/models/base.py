from __future__ import annotations

from typing import Protocol

from app.schemas.api import NormalizedAISRequest
from app.schemas.responses import AISModelResponse


class BaseResponsesAdapter(Protocol):
    def create_response(self, request: NormalizedAISRequest) -> AISModelResponse:
        ...


class AdapterUnavailableError(RuntimeError):
    pass
