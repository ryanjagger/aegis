from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db import models, repository


@dataclass(frozen=True)
class NimbusLedgerUpdate:
    score_delta: float
    score_total: float
    budget: float
    zone: str
    policy_action: str
    reason: str


def zone_for_score(score_total: float, budget: float = 10.0) -> str:
    ratio = score_total / budget if budget else 0.0
    if ratio >= 1.0:
        return "BLOCK"
    if ratio >= 0.8:
        return "SANITIZE"
    if ratio >= 0.6:
        return "WARN"
    return "PASS"


def policy_action_for_zone(zone: str) -> str:
    if zone == "PASS":
        return "ALLOW"
    return zone


def latest_score_total(db: Session, session_id: str) -> float:
    stmt = (
        select(models.LeakageLedgerRecord)
        .where(models.LeakageLedgerRecord.session_id == session_id)
        .order_by(
            models.LeakageLedgerRecord.turn_id.desc(),
            models.LeakageLedgerRecord.created_at.desc(),
        )
        .limit(1)
    )
    row = db.execute(stmt).scalar_one_or_none()
    return float(row.score_total) if row is not None else 0.0


def record_score_update(
    db: Session,
    *,
    session_id: str,
    request_id: str,
    response_id: str,
    turn_id: int,
    score_delta: float,
    reasons: list[str],
    budget: float = 10.0,
) -> NimbusLedgerUpdate:
    previous_total = latest_score_total(db, session_id)
    score_total = round(previous_total + score_delta, 2)
    zone = zone_for_score(score_total, budget=budget)
    reason = "; ".join(reasons) if reasons else "no risk features"
    db.add(
        models.LeakageLedgerRecord(
            id=repository.short_id("led"),
            session_id=session_id,
            request_id=request_id,
            response_id=response_id,
            turn_id=turn_id,
            score_delta=score_delta,
            score_total=score_total,
            budget=budget,
            zone=zone,
            reason=reason,
        )
    )
    return NimbusLedgerUpdate(
        score_delta=score_delta,
        score_total=score_total,
        budget=budget,
        zone=zone,
        policy_action=policy_action_for_zone(zone),
        reason=reason,
    )
