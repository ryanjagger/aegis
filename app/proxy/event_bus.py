from __future__ import annotations

from typing import Any

from sqlalchemy.orm import Session

from app.db import repository


class EventBus:
    def __init__(self, db: Session, *, request_id: str, session_id: str) -> None:
        self.db = db
        self.request_id = request_id
        self.session_id = session_id

    def emit(
        self, event_type: str, payload: dict[str, Any], response_id: str | None = None
    ) -> None:
        repository.add_event(
            self.db,
            request_id=self.request_id,
            response_id=response_id,
            session_id=self.session_id,
            event_type=event_type,
            payload=payload,
        )
