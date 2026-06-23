from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import CanaryRecord


def list_canaries_for_request(db: Session, request_id: str) -> list[CanaryRecord]:
    return list(
        db.execute(select(CanaryRecord).where(CanaryRecord.request_id == request_id)).scalars()
    )


def list_canaries_for_session(db: Session, session_id: str) -> list[CanaryRecord]:
    return list(
        db.execute(select(CanaryRecord).where(CanaryRecord.session_id == session_id)).scalars()
    )
