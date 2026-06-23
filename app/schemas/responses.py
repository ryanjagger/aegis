from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class AISModelResponse(BaseModel):
    id: str
    model: str
    output: list[dict[str, Any]]
    status: str = "completed"
    provider_response_id: str | None = None

    model_config = ConfigDict(extra="allow")


class ResponsesCreateRequest(BaseModel):
    model: str = "mock-ais"
    instructions: str | None = None
    input: str | list[dict[str, Any]] = Field(default_factory=list)
    tools: list[dict[str, Any]] = Field(default_factory=list)
    store: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="allow")
