from __future__ import annotations

import json
from typing import Any
from uuid import uuid4

from sqlalchemy import Select, func, select
from sqlalchemy.orm import Session

from app.db import models
from app.db.models import utc_now_iso
from app.schemas.events import DetectorHit


def short_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:12]}"


def dumps_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def loads_json(value: str | None) -> Any:
    if not value:
        return None
    return json.loads(value)


def _row_to_dict(row: Any) -> dict[str, Any]:
    data = {column.name: getattr(row, column.name) for column in row.__table__.columns}
    for key, value in list(data.items()):
        if key.endswith("_json"):
            data[key] = loads_json(value)
    return data


def next_turn_id(db: Session, session_id: str) -> int:
    stmt = select(func.max(models.RequestRecord.turn_id)).where(
        models.RequestRecord.session_id == session_id
    )
    current = db.execute(stmt).scalar_one_or_none()
    return int(current or 0) + 1


def add_event(
    db: Session,
    *,
    request_id: str,
    session_id: str,
    event_type: str,
    payload: dict[str, Any],
    response_id: str | None = None,
) -> models.EventRecord:
    event = models.EventRecord(
        id=short_id("evt"),
        request_id=request_id,
        response_id=response_id,
        session_id=session_id,
        event_type=event_type,
        payload_json=dumps_json(payload),
    )
    db.add(event)
    return event


def add_detector_hit(
    db: Session,
    *,
    request_id: str,
    response_id: str,
    session_id: str,
    hit: DetectorHit,
) -> models.DetectorEventRecord:
    row = models.DetectorEventRecord(
        id=short_id("det"),
        request_id=request_id,
        response_id=response_id,
        session_id=session_id,
        detector=hit.detector,
        surface=hit.surface,
        severity=hit.severity,
        matched_canary_id=hit.matched_canary_id,
        evidence_preview=hit.evidence_preview,
        policy_recommendation=hit.policy_recommendation,
    )
    db.add(row)
    return row


def add_response_item(
    db: Session,
    *,
    response_id: str,
    request_id: str,
    session_id: str,
    item_index: int,
    item: dict[str, Any],
) -> models.ResponseItemRecord:
    row = models.ResponseItemRecord(
        id=short_id("item"),
        response_id=response_id,
        request_id=request_id,
        session_id=session_id,
        item_index=item_index,
        item_type=item.get("type"),
        role=item.get("role"),
        surface_path=f"response.output[{item_index}]",
        item_json=dumps_json(item),
    )
    db.add(row)
    return row


def add_tool_call(
    db: Session,
    *,
    request_id: str,
    response_id: str,
    session_id: str,
    call_id: str,
    tool_name: str,
    arguments: Any,
    allowed: bool,
    executed: bool,
    result: Any = None,
    block_reason: str | None = None,
) -> models.ToolCallRecord:
    row = models.ToolCallRecord(
        id=short_id("tool"),
        request_id=request_id,
        response_id=response_id,
        session_id=session_id,
        call_id=call_id,
        tool_name=tool_name,
        arguments_json=dumps_json(arguments),
        allowed=1 if allowed else 0,
        executed=1 if executed else 0,
        result_json=dumps_json(result) if result is not None else None,
        block_reason=block_reason,
    )
    db.add(row)
    return row


def mark_canary_leaked(
    db: Session, *, canary_id: str, response_id: str, surface: str
) -> models.CanaryRecord | None:
    canary = db.get(models.CanaryRecord, canary_id)
    if canary is None:
        return None
    canary.response_id = response_id
    canary.leaked = 1
    canary.first_leaked_at = canary.first_leaked_at or utc_now_iso()
    canary.leaked_surface = surface
    return canary


def set_canary_response_id(db: Session, *, request_id: str, response_id: str) -> None:
    rows = db.execute(
        select(models.CanaryRecord).where(models.CanaryRecord.request_id == request_id)
    ).scalars()
    for row in rows:
        row.response_id = response_id


def add_canary(
    db: Session,
    *,
    canary_id: str,
    session_id: str,
    request_id: str,
    value: str,
    value_hash: str,
    format: str,
    source_label: str,
    response_id: str | None = None,
) -> models.CanaryRecord:
    row = models.CanaryRecord(
        id=canary_id,
        session_id=session_id,
        request_id=request_id,
        response_id=response_id,
        value=value,
        value_hash=value_hash,
        format=format,
        source_label=source_label,
    )
    db.add(row)
    return row


def add_response_record(
    db: Session,
    *,
    response_id: str,
    provider_response_id: str | None,
    request_id: str,
    session_id: str,
    turn_id: int,
    route: str,
    model: str,
    scenario: str | None,
    raw_request: dict[str, Any],
    normalized_input: list[dict[str, Any]],
    injected_context: str | None,
    raw_response: dict[str, Any],
    final_response: dict[str, Any],
    output_text: str | None,
    policy_action: str,
    status: str,
    latency_ms: int,
) -> models.ResponseRecord:
    row = models.ResponseRecord(
        id=response_id,
        provider_response_id=provider_response_id,
        request_id=request_id,
        session_id=session_id,
        turn_id=turn_id,
        route=route,
        model=model,
        scenario=scenario,
        raw_request_json=dumps_json(raw_request),
        normalized_input_json=dumps_json(normalized_input),
        injected_context=injected_context,
        raw_response_json=dumps_json(raw_response),
        final_response_json=dumps_json(final_response),
        output_text=output_text,
        policy_action=policy_action,
        status=status,
        latency_ms=latency_ms,
    )
    db.add(row)
    return row


def add_request_record(
    db: Session,
    *,
    request_id: str,
    session_id: str,
    turn_id: int,
    route: str,
    scenario: str | None,
    user_input: str | None,
    injected_context: str | None,
    raw_output: str | None,
    final_output: str | None,
    policy_action: str,
    status: str,
    latency_ms: int,
) -> models.RequestRecord:
    row = models.RequestRecord(
        id=request_id,
        session_id=session_id,
        turn_id=turn_id,
        route=route,
        scenario=scenario,
        user_input=user_input,
        injected_context=injected_context,
        raw_output=raw_output,
        final_output=final_output,
        policy_action=policy_action,
        status=status,
        latency_ms=latency_ms,
    )
    db.add(row)
    return row


def list_rows(db: Session, model: type[Any], *, limit: int = 200) -> list[dict[str, Any]]:
    stmt: Select[Any] = select(model).order_by(model.created_at.desc()).limit(limit)
    return [_row_to_dict(row) for row in db.execute(stmt).scalars()]


def get_row(db: Session, model: type[Any], row_id: str) -> dict[str, Any] | None:
    row = db.get(model, row_id)
    return _row_to_dict(row) if row is not None else None


def list_rows_for_request(
    db: Session, model: type[Any], *, request_id: str, limit: int = 200
) -> list[dict[str, Any]]:
    stmt = (
        select(model)
        .where(model.request_id == request_id)
        .order_by(model.created_at.desc())
        .limit(limit)
    )
    return [_row_to_dict(row) for row in db.execute(stmt).scalars()]


def list_rows_for_response(
    db: Session, model: type[Any], *, response_id: str, limit: int = 200
) -> list[dict[str, Any]]:
    stmt = (
        select(model)
        .where(model.response_id == response_id)
        .order_by(model.created_at.desc())
        .limit(limit)
    )
    return [_row_to_dict(row) for row in db.execute(stmt).scalars()]


def clear_all(db: Session) -> None:
    for model in (
        models.EventRecord,
        models.DetectorEventRecord,
        models.ToolCallRecord,
        models.ResponseItemRecord,
        models.LeakageLedgerRecord,
        models.CanaryRecord,
        models.ResponseRecord,
        models.RequestRecord,
    ):
        db.query(model).delete()
    db.commit()
