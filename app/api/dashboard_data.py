from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db import models, repository
from app.db.database import get_db

router = APIRouter()
DbSession = Annotated[Session, Depends(get_db)]


@router.get("/requests")
def list_requests(db: DbSession) -> list[dict]:
    return repository.list_rows(db, models.RequestRecord)


@router.get("/requests/{request_id}")
def get_request(request_id: str, db: DbSession) -> dict:
    row = repository.get_row(db, models.RequestRecord, request_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Request not found")
    row["events"] = repository.list_rows_for_request(db, models.EventRecord, request_id=request_id)
    row["detector_events"] = repository.list_rows_for_request(
        db, models.DetectorEventRecord, request_id=request_id
    )
    row["canaries"] = repository.list_rows_for_request(
        db, models.CanaryRecord, request_id=request_id
    )
    row["tool_calls"] = repository.list_rows_for_request(
        db, models.ToolCallRecord, request_id=request_id
    )
    return row


@router.get("/responses")
def list_responses(db: DbSession) -> list[dict]:
    return repository.list_rows(db, models.ResponseRecord)


@router.get("/responses/{response_id}")
def get_response(response_id: str, db: DbSession) -> dict:
    row = repository.get_row(db, models.ResponseRecord, response_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Response not found")
    row["response_items"] = repository.list_rows_for_response(
        db, models.ResponseItemRecord, response_id=response_id
    )
    row["events"] = repository.list_rows_for_response(
        db, models.EventRecord, response_id=response_id
    )
    row["detector_events"] = repository.list_rows_for_response(
        db, models.DetectorEventRecord, response_id=response_id
    )
    row["canaries"] = repository.list_rows_for_response(
        db, models.CanaryRecord, response_id=response_id
    )
    row["tool_calls"] = repository.list_rows_for_response(
        db, models.ToolCallRecord, response_id=response_id
    )
    return row


@router.get("/canaries")
def list_canaries(db: DbSession) -> list[dict]:
    return repository.list_rows(db, models.CanaryRecord)


@router.get("/events")
def list_events(db: DbSession) -> list[dict]:
    return repository.list_rows(db, models.EventRecord)


@router.get("/detector-events")
def list_detector_events(db: DbSession) -> list[dict]:
    return repository.list_rows(db, models.DetectorEventRecord)


@router.get("/tool-calls")
def list_tool_calls(db: DbSession) -> list[dict]:
    return repository.list_rows(db, models.ToolCallRecord)


@router.get("/leakage-ledger")
def list_leakage_ledger(db: DbSession) -> list[dict]:
    return repository.list_rows(db, models.LeakageLedgerRecord)
