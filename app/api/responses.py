from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Body, Depends, Request
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.proxy.responses_proxy import run_responses_pipeline

router = APIRouter()


@router.post("/v1/responses")
def create_response(
    request: Request,
    body: Annotated[dict[str, Any], Body(default_factory=dict)],
    db: Annotated[Session, Depends(get_db)],
) -> dict[str, Any]:
    return run_responses_pipeline(
        db,
        body=body,
        headers=dict(request.headers),
        route="/v1/responses",
    )
