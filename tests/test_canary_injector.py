from sqlalchemy import select

from app.canaries.injector import inject_canaries
from app.db.database import SessionLocal
from app.db.models import CanaryRecord, EventRecord


def test_canary_injection_registers_canaries() -> None:
    with SessionLocal() as db:
        items, context, canaries = inject_canaries(
            db,
            input_items=[{"role": "user", "content": "hello"}],
            request_id="req_test",
            session_id="sess_test",
        )
        db.commit()

        rows = list(db.execute(select(CanaryRecord)).scalars())
        events = list(db.execute(select(EventRecord)).scalars())

    assert len(canaries) == 5
    assert len(rows) == 5
    assert len(events) == 5
    assert "Internal diagnostic appendix" in context
    assert items[-1]["content"] == context
