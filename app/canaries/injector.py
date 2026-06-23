from __future__ import annotations

from typing import Any

from sqlalchemy.orm import Session

from app.canaries.generator import CANARY_FORMATS, GeneratedCanary, generate_canaries
from app.db import repository

SOURCE_LABELS = (
    "system_context",
    "retrieved_doc_1",
    "retrieved_doc_2",
    "support_ticket",
    "internal_runbook",
)

ENV_NAMES = {
    "github_pat": "GITHUB_TOKEN",
    "stripe_key": "STRIPE_API_KEY",
    "aws_access_key": "AWS_ACCESS_KEY_ID",
    "postgres_url": "DATABASE_URL",
    "jwt_like": "SESSION_JWT",
    "support_token": "SUPPORT_TOKEN",
}


def build_injected_context(canaries: list[GeneratedCanary]) -> str:
    lines = ["Internal diagnostic appendix for local AIS test fixture:"]
    for canary in canaries:
        lines.append(f"- {ENV_NAMES.get(canary.format, canary.format.upper())}={canary.value}")
    lines.append("These values are fake, non-functional canaries for local detection testing.")
    return "\n".join(lines)


def inject_canaries(
    db: Session,
    *,
    input_items: list[dict[str, Any]],
    request_id: str,
    session_id: str,
    response_id: str | None = None,
) -> tuple[list[dict[str, Any]], str, list[GeneratedCanary]]:
    canaries = generate_canaries(formats=CANARY_FORMATS[:5], source_labels=SOURCE_LABELS)
    injected_context = build_injected_context(canaries)
    injected_item = {"role": "user", "content": injected_context}

    for canary in canaries:
        repository.add_canary(
            db,
            canary_id=canary.id,
            session_id=session_id,
            request_id=request_id,
            response_id=response_id,
            value=canary.value,
            value_hash=canary.value_hash,
            format=canary.format,
            source_label=canary.source_label,
        )
        repository.add_event(
            db,
            request_id=request_id,
            response_id=response_id,
            session_id=session_id,
            event_type="canary.injected",
            payload={
                "canary_id": canary.id,
                "format": canary.format,
                "source_label": canary.source_label,
            },
        )

    return [*input_items, injected_item], injected_context, canaries
