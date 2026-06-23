from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import Integer, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


class Base(DeclarativeBase):
    pass


class ResponseRecord(Base):
    __tablename__ = "responses"

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    provider_response_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    request_id: Mapped[str] = mapped_column(Text, index=True)
    session_id: Mapped[str] = mapped_column(Text, index=True)
    turn_id: Mapped[int] = mapped_column(Integer, index=True)
    route: Mapped[str] = mapped_column(Text)
    model: Mapped[str] = mapped_column(Text)
    scenario: Mapped[str | None] = mapped_column(Text, nullable=True)
    raw_request_json: Mapped[str] = mapped_column(Text)
    normalized_input_json: Mapped[str] = mapped_column(Text)
    injected_context: Mapped[str | None] = mapped_column(Text, nullable=True)
    raw_response_json: Mapped[str] = mapped_column(Text)
    final_response_json: Mapped[str] = mapped_column(Text)
    output_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    policy_action: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(Text)
    latency_ms: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[str] = mapped_column(Text, default=utc_now_iso)


class ResponseItemRecord(Base):
    __tablename__ = "response_items"

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    response_id: Mapped[str] = mapped_column(Text, index=True)
    request_id: Mapped[str] = mapped_column(Text, index=True)
    session_id: Mapped[str] = mapped_column(Text, index=True)
    item_index: Mapped[int] = mapped_column(Integer)
    item_type: Mapped[str | None] = mapped_column(Text, nullable=True)
    role: Mapped[str | None] = mapped_column(Text, nullable=True)
    surface_path: Mapped[str] = mapped_column(Text)
    item_json: Mapped[str] = mapped_column(Text)
    created_at: Mapped[str] = mapped_column(Text, default=utc_now_iso)


class RequestRecord(Base):
    __tablename__ = "requests"

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    session_id: Mapped[str] = mapped_column(Text, index=True)
    turn_id: Mapped[int] = mapped_column(Integer, index=True)
    route: Mapped[str] = mapped_column(Text)
    scenario: Mapped[str | None] = mapped_column(Text, nullable=True)
    user_input: Mapped[str | None] = mapped_column(Text, nullable=True)
    injected_context: Mapped[str | None] = mapped_column(Text, nullable=True)
    raw_output: Mapped[str | None] = mapped_column(Text, nullable=True)
    final_output: Mapped[str | None] = mapped_column(Text, nullable=True)
    policy_action: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(Text)
    latency_ms: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[str] = mapped_column(Text, default=utc_now_iso)


class CanaryRecord(Base):
    __tablename__ = "canaries"

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    session_id: Mapped[str] = mapped_column(Text, index=True)
    request_id: Mapped[str] = mapped_column(Text, index=True)
    response_id: Mapped[str | None] = mapped_column(Text, nullable=True, index=True)
    value: Mapped[str] = mapped_column(Text)
    value_hash: Mapped[str] = mapped_column(Text)
    format: Mapped[str] = mapped_column(Text)
    source_label: Mapped[str] = mapped_column(Text)
    leaked: Mapped[int] = mapped_column(Integer, default=0)
    first_leaked_at: Mapped[str | None] = mapped_column(Text, nullable=True)
    leaked_surface: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[str] = mapped_column(Text, default=utc_now_iso)


class DetectorEventRecord(Base):
    __tablename__ = "detector_events"

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    request_id: Mapped[str] = mapped_column(Text, index=True)
    response_id: Mapped[str] = mapped_column(Text, index=True)
    session_id: Mapped[str] = mapped_column(Text, index=True)
    detector: Mapped[str] = mapped_column(Text)
    surface: Mapped[str] = mapped_column(Text)
    severity: Mapped[str] = mapped_column(Text)
    matched_canary_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    evidence_preview: Mapped[str] = mapped_column(Text)
    policy_recommendation: Mapped[str] = mapped_column(Text)
    created_at: Mapped[str] = mapped_column(Text, default=utc_now_iso)


class ToolCallRecord(Base):
    __tablename__ = "tool_calls"

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    request_id: Mapped[str] = mapped_column(Text, index=True)
    response_id: Mapped[str] = mapped_column(Text, index=True)
    session_id: Mapped[str] = mapped_column(Text, index=True)
    call_id: Mapped[str] = mapped_column(Text)
    tool_name: Mapped[str] = mapped_column(Text)
    arguments_json: Mapped[str] = mapped_column(Text)
    allowed: Mapped[int] = mapped_column(Integer)
    executed: Mapped[int] = mapped_column(Integer)
    result_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    block_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[str] = mapped_column(Text, default=utc_now_iso)


class LeakageLedgerRecord(Base):
    __tablename__ = "leakage_ledger"

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    session_id: Mapped[str] = mapped_column(Text, index=True)
    request_id: Mapped[str] = mapped_column(Text, index=True)
    response_id: Mapped[str] = mapped_column(Text, index=True)
    turn_id: Mapped[int] = mapped_column(Integer)
    score_delta: Mapped[float] = mapped_column(default=0.0)
    score_total: Mapped[float] = mapped_column(default=0.0)
    budget: Mapped[float] = mapped_column(default=10.0)
    zone: Mapped[str] = mapped_column(Text, default="PASS")
    reason: Mapped[str] = mapped_column(Text, default="NIMBUS-lite disabled")
    created_at: Mapped[str] = mapped_column(Text, default=utc_now_iso)


class EventRecord(Base):
    __tablename__ = "events"

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    request_id: Mapped[str] = mapped_column(Text, index=True)
    response_id: Mapped[str | None] = mapped_column(Text, nullable=True, index=True)
    session_id: Mapped[str] = mapped_column(Text, index=True)
    event_type: Mapped[str] = mapped_column(Text, index=True)
    payload_json: Mapped[str] = mapped_column(Text)
    created_at: Mapped[str] = mapped_column(Text, default=utc_now_iso)
