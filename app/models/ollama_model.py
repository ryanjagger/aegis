from __future__ import annotations

from app.models.base import AdapterUnavailableError
from app.schemas.api import NormalizedAISRequest
from app.schemas.responses import AISModelResponse


class OllamaAdapter:
    def create_response(self, request: NormalizedAISRequest) -> AISModelResponse:
        raise AdapterUnavailableError("OllamaAdapter is a placeholder for a later phase.")
