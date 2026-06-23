from __future__ import annotations

from app.models.base import AdapterUnavailableError
from app.schemas.api import NormalizedAISRequest
from app.schemas.responses import AISModelResponse


class OpenAIResponsesAdapter:
    def create_response(self, request: NormalizedAISRequest) -> AISModelResponse:
        raise AdapterUnavailableError(
            "OpenAIResponsesAdapter is intentionally stubbed in this local AIS demo."
        )
