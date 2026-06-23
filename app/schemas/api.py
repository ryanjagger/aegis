from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class DefenseConfig(BaseModel):
    canary_injection: bool = True
    output_scanning: bool = True
    tool_scanning: bool = True
    nimbus_lite: bool = False


class ChatRequest(BaseModel):
    session_id: str | None = None
    user_input: str
    scenario: str = "benign"
    defenses: DefenseConfig = Field(default_factory=DefenseConfig)
    model_adapter: str = "mock"


class PlaygroundRunRequest(BaseModel):
    route: Literal["/chat", "/v1/responses"] = "/chat"
    session_id: str | None = None
    user_input: str = "Summarize this support ticket and create an internal note."
    scenario: str = "benign"
    defenses: DefenseConfig = Field(default_factory=DefenseConfig)
    model_adapter: str = "mock"


class NormalizedAISRequest(BaseModel):
    request_id: str
    response_id: str | None = None
    session_id: str
    turn_id: int
    route: Literal["/v1/responses", "/chat", "/playground/run"]
    model: str
    instructions: str | None = None
    input_items: list[dict[str, Any]]
    tools: list[dict[str, Any]] = Field(default_factory=list)
    scenario: str | None = None
    store: bool = False
    defenses: DefenseConfig = Field(default_factory=DefenseConfig)
    raw_request_json: dict[str, Any] = Field(default_factory=dict)
    injected_context: str | None = None
    model_adapter: str = "mock"


class DashboardResponse(BaseModel):
    model_config = ConfigDict(extra="allow")
