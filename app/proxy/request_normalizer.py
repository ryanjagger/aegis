from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from app.config import get_settings
from app.db import repository
from app.schemas.api import DefenseConfig, NormalizedAISRequest


def _truthy(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return default


def _header(headers: Mapping[str, str], *names: str) -> str | None:
    lowered = {key.lower(): value for key, value in headers.items()}
    for name in names:
        value = lowered.get(name.lower())
        if value is not None:
            return value
    return None


def _metadata_bool(metadata: dict[str, Any], *names: str, default: bool) -> bool:
    for name in names:
        if name in metadata:
            return _truthy(metadata[name], default)
    return default


def extract_defenses(headers: Mapping[str, str], metadata: dict[str, Any]) -> DefenseConfig:
    return DefenseConfig(
        canary_injection=_truthy(
            _header(headers, "X-AIS-Canary-Injection", "X-AISIO-Canary-Injection"),
            _metadata_bool(
                metadata,
                "ais_canary_injection",
                "aisio_canary_injection",
                default=True,
            ),
        ),
        output_scanning=_truthy(
            _header(headers, "X-AIS-Output-Scanning", "X-AISIO-Output-Scanning"),
            _metadata_bool(
                metadata,
                "ais_output_scanning",
                "aisio_output_scanning",
                default=True,
            ),
        ),
        tool_scanning=_truthy(
            _header(headers, "X-AIS-Tool-Scanning", "X-AISIO-Tool-Scanning"),
            _metadata_bool(
                metadata,
                "ais_tool_scanning",
                "aisio_tool_scanning",
                default=True,
            ),
        ),
        nimbus_lite=_truthy(
            _header(headers, "X-AIS-Nimbus-Lite", "X-AISIO-Nimbus-Lite"),
            _metadata_bool(metadata, "ais_nimbus_lite", "aisio_nimbus_lite", default=False),
        ),
    )


def normalize_request(
    *,
    body: dict[str, Any],
    headers: Mapping[str, str],
    route: str,
    request_id: str,
    response_id: str,
    turn_id: int,
) -> NormalizedAISRequest:
    settings = get_settings()
    metadata = dict(body.get("metadata") or {})
    raw_input = body.get("input", [])
    if isinstance(raw_input, str):
        input_items = [{"role": "user", "content": raw_input}]
    elif isinstance(raw_input, list):
        input_items = [
            item if isinstance(item, dict) else {"role": "user", "content": item}
            for item in raw_input
        ]
    else:
        input_items = [{"role": "user", "content": str(raw_input)}]

    session_id = (
        metadata.get("session_id")
        or _header(headers, "X-AIS-Session-ID", "X-AISIO-Session-ID")
        or repository.short_id("sess")
    )
    scenario = metadata.get("scenario") or body.get("scenario")
    model_adapter = metadata.get("model_adapter") or body.get("model_adapter") or "mock"
    store = bool(body.get("store", False))
    if not settings.allow_provider_store:
        store = False

    return NormalizedAISRequest(
        request_id=request_id,
        response_id=response_id,
        session_id=str(session_id),
        turn_id=turn_id,
        route=route,  # type: ignore[arg-type]
        model=str(body.get("model") or settings.default_model),
        instructions=body.get("instructions"),
        input_items=input_items,
        tools=list(body.get("tools") or []),
        scenario=str(scenario) if scenario else None,
        store=store,
        defenses=extract_defenses(headers, metadata),
        raw_request_json=body,
        model_adapter=str(model_adapter),
    )
