from __future__ import annotations

import json
import time
from typing import Any

from sqlalchemy.orm import Session

from app.canaries.injector import inject_canaries
from app.canaries.registry import list_canaries_for_request
from app.db import repository
from app.models.base import AdapterUnavailableError, BaseResponsesAdapter
from app.models.mock_responses import MockResponsesAdapter
from app.models.ollama_model import OllamaAdapter
from app.models.openai_responses import OpenAIResponsesAdapter
from app.nimbus.ledger import NimbusLedgerUpdate, record_score_update
from app.nimbus.scoring import score_request
from app.proxy.event_bus import EventBus
from app.proxy.policy import apply_policy
from app.proxy.request_normalizer import normalize_request
from app.scanners.canary_scanner import CanaryScanner
from app.scanners.responses_item_scanner import ResponsesItemScanner
from app.schemas.api import NormalizedAISRequest
from app.schemas.events import DetectorHit
from app.tools.tool_proxy import ToolProxy, ToolProxyResult


def get_adapter(name: str) -> BaseResponsesAdapter:
    normalized = name.strip().lower()
    if normalized in {"mock", "mock-ais", "mock-aisio"}:
        return MockResponsesAdapter()
    if normalized in {"openai", "openai_responses"}:
        return OpenAIResponsesAdapter()
    if normalized == "ollama":
        return OllamaAdapter()
    raise AdapterUnavailableError(f"Unknown model adapter: {name}")


def run_responses_pipeline(
    db: Session,
    *,
    body: dict[str, Any],
    headers: dict[str, str],
    route: str,
    user_input: str | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    request_id = repository.short_id("req")
    response_id = repository.short_id("resp")

    preliminary_session = str(
        (body.get("metadata") or {}).get("session_id") or headers.get("x-ais-session-id") or ""
    )
    turn_id = repository.next_turn_id(db, preliminary_session) if preliminary_session else 1

    normalized = normalize_request(
        body=body,
        headers=headers,
        route=route,
        request_id=request_id,
        response_id=response_id,
        turn_id=turn_id,
    )
    if not preliminary_session:
        normalized.turn_id = repository.next_turn_id(db, normalized.session_id)

    bus = EventBus(db, request_id=request_id, session_id=normalized.session_id)
    bus.emit(
        "request.received",
        {"route": route, "model": normalized.model, "scenario": normalized.scenario},
        response_id=response_id,
    )

    if normalized.defenses.canary_injection:
        input_items, injected_context, _canaries = inject_canaries(
            db,
            input_items=normalized.input_items,
            request_id=request_id,
            session_id=normalized.session_id,
            response_id=response_id,
            source=normalized.defenses.canary_source,
        )
        normalized.input_items = input_items
        normalized.injected_context = injected_context
        db.flush()
    else:
        normalized.injected_context = None

    try:
        adapter = get_adapter(normalized.model_adapter)
        model_response = adapter.create_response(normalized)
    except AdapterUnavailableError as exc:
        model_response = MockResponsesAdapter().create_response(
            normalized.model_copy(update={"scenario": "benign"})
        )
        model_response.output = [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": f"[ADAPTER UNAVAILABLE: {exc}]"}],
            }
        ]

    bus.emit(
        "model.response.created",
        {"output_item_count": len(model_response.output), "adapter": normalized.model_adapter},
        response_id=response_id,
    )

    for index, item in enumerate(model_response.output):
        repository.add_response_item(
            db,
            response_id=response_id,
            request_id=request_id,
            session_id=normalized.session_id,
            item_index=index,
            item=item,
        )

    output_hits: list[DetectorHit] = []
    tool_results: list[ToolProxyResult] = []
    db.flush()
    canaries = list_canaries_for_request(db, request_id)
    if normalized.defenses.output_scanning:
        scanner = ResponsesItemScanner(CanaryScanner(canaries))
        output_hits = scanner.scan_output(
            model_response.output,
            scan_messages=True,
            scan_tool_items=False,
            scan_serialized_output=False,
        )
        _persist_output_hits(
            db,
            bus=bus,
            request=normalized,
            hits=output_hits,
        )

    tool_results = _handle_function_calls(
        db,
        request=normalized,
        output=model_response.output,
        canary_scanner=CanaryScanner(canaries),
    )
    all_hits = [*output_hits, *[hit for result in tool_results for hit in result.hits]]

    nimbus_update = _handle_nimbus_lite(
        db,
        bus=bus,
        request=normalized,
        output=model_response.output,
        hits=all_hits,
        canaries=canaries,
    )

    decision = apply_policy(
        model_response.output,
        all_hits,
        nimbus_action=nimbus_update.policy_action if nimbus_update else None,
    )
    repository.set_canary_response_id(db, request_id=request_id, response_id=response_id)

    final_response = _build_response_object(
        normalized=normalized,
        output=decision.output,
        status=model_response.status,
        policy_action=decision.action,
        detector_hit_count=len(all_hits),
        tool_results=tool_results,
        nimbus_update=nimbus_update,
    )
    raw_response = _build_response_object(
        normalized=normalized,
        output=model_response.output,
        status=model_response.status,
        policy_action="RAW",
        detector_hit_count=0,
        tool_results=tool_results,
        nimbus_update=nimbus_update,
    )

    latency_ms = int((time.perf_counter() - started) * 1000)
    repository.add_response_record(
        db,
        response_id=response_id,
        provider_response_id=model_response.provider_response_id,
        request_id=request_id,
        session_id=normalized.session_id,
        turn_id=normalized.turn_id,
        route=route,
        model=normalized.model,
        scenario=normalized.scenario,
        raw_request=body,
        normalized_input=normalized.input_items,
        injected_context=normalized.injected_context,
        raw_response=raw_response,
        final_response=final_response,
        output_text=decision.output_text,
        policy_action=decision.action,
        status=model_response.status,
        latency_ms=latency_ms,
    )
    repository.add_request_record(
        db,
        request_id=request_id,
        session_id=normalized.session_id,
        turn_id=normalized.turn_id,
        route=route,
        scenario=normalized.scenario,
        user_input=user_input or _extract_user_input(normalized),
        injected_context=normalized.injected_context,
        raw_output=json.dumps(model_response.output, sort_keys=True),
        final_output=decision.output_text,
        policy_action=decision.action,
        status=model_response.status,
        latency_ms=latency_ms,
    )
    bus.emit(
        "policy.applied",
        {"policy_action": decision.action, "detector_hit_count": len(all_hits)},
        response_id=response_id,
    )
    bus.emit("response.returned", {"status": model_response.status}, response_id=response_id)
    db.commit()
    return final_response


def _build_response_object(
    *,
    normalized: NormalizedAISRequest,
    output: list[dict[str, Any]],
    status: str,
    policy_action: str,
    detector_hit_count: int,
    tool_results: list[ToolProxyResult] | None = None,
    nimbus_update: NimbusLedgerUpdate | None = None,
) -> dict[str, Any]:
    tool_results = tool_results or []
    tool_call_count = len(tool_results)
    tool_blocked_count = sum(1 for result in tool_results if not result.allowed)
    tool_executed_count = sum(1 for result in tool_results if result.executed)
    metadata = {
        "ais_request_id": normalized.request_id,
        "ais_session_id": normalized.session_id,
        "ais_policy_action": policy_action,
        "ais_detector_hit_count": str(detector_hit_count),
        "ais_tool_call_count": str(tool_call_count),
        "ais_tool_blocked_count": str(tool_blocked_count),
        "ais_tool_executed_count": str(tool_executed_count),
    }
    if nimbus_update:
        metadata.update(
            {
                "ais_nimbus_score_delta": f"{nimbus_update.score_delta:.2f}",
                "ais_nimbus_score_total": f"{nimbus_update.score_total:.2f}",
                "ais_nimbus_budget": f"{nimbus_update.budget:.2f}",
                "ais_nimbus_zone": nimbus_update.zone,
            }
        )
    return {
        "id": normalized.response_id,
        "object": "response",
        "created_at": int(time.time()),
        "model": normalized.model,
        "output": output,
        "status": status,
        "metadata": metadata,
    }


def _extract_user_input(normalized: NormalizedAISRequest) -> str | None:
    for item in normalized.input_items:
        if item.get("role") != "user":
            continue
        content = item.get("content")
        if isinstance(content, str) and not content.startswith("Internal diagnostic appendix"):
            return content
    return None


def _persist_output_hits(
    db: Session,
    *,
    bus: EventBus,
    request: NormalizedAISRequest,
    hits: list[DetectorHit],
) -> None:
    for hit in hits:
        repository.add_detector_hit(
            db,
            request_id=request.request_id,
            response_id=request.response_id or "",
            session_id=request.session_id,
            hit=hit,
        )
        if hit.matched_canary_id:
            repository.mark_canary_leaked(
                db,
                canary_id=hit.matched_canary_id,
                response_id=request.response_id or "",
                surface=hit.surface,
            )
        bus.emit(
            "detector.hit",
            {
                "detector": hit.detector,
                "surface": hit.surface,
                "severity": hit.severity,
                "matched_canary_id": hit.matched_canary_id,
            },
            response_id=request.response_id,
        )


def _handle_nimbus_lite(
    db: Session,
    *,
    bus: EventBus,
    request: NormalizedAISRequest,
    output: list[dict[str, Any]],
    hits: list[DetectorHit],
    canaries: list[Any],
) -> NimbusLedgerUpdate | None:
    if not request.defenses.nimbus_lite:
        return None

    score = score_request(request=request, output=output, hits=hits, canaries=canaries)
    update = record_score_update(
        db,
        session_id=request.session_id,
        request_id=request.request_id,
        response_id=request.response_id or "",
        turn_id=request.turn_id,
        score_delta=score.score_delta,
        reasons=score.reasons,
    )
    bus.emit(
        "nimbus.scored",
        {
            "score_delta": update.score_delta,
            "score_total": update.score_total,
            "budget": update.budget,
            "zone": update.zone,
            "policy_action": update.policy_action,
            "reason": update.reason,
        },
        response_id=request.response_id,
    )
    return update


def _handle_function_calls(
    db: Session,
    *,
    request: NormalizedAISRequest,
    output: list[dict[str, Any]],
    canary_scanner: CanaryScanner,
) -> list[ToolProxyResult]:
    proxy = ToolProxy(
        db,
        request_id=request.request_id,
        response_id=request.response_id or "",
        session_id=request.session_id,
        canary_scanner=canary_scanner,
        scan_enabled=request.defenses.tool_scanning,
    )
    results: list[ToolProxyResult] = []
    for index, item in enumerate(output):
        if item.get("type") != "function_call":
            continue
        results.append(proxy.handle_function_call(item, item_index=index))
    return results
