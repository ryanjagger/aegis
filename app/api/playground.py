from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Request
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db.database import get_db
from app.proxy.responses_proxy import run_responses_pipeline
from app.schemas.api import PlaygroundRunRequest

router = APIRouter()


@router.post("/playground/run")
def run_playground(
    request: Request,
    payload: PlaygroundRunRequest,
    db: Annotated[Session, Depends(get_db)],
) -> dict:
    settings = get_settings()
    body = {
        "model": settings.default_model,
        "input": [{"role": "user", "content": payload.user_input}],
        "store": False,
        "metadata": {
            "session_id": payload.session_id,
            "scenario": payload.scenario,
            "model_adapter": payload.model_adapter,
            "ais_canary_injection": payload.defenses.canary_injection,
            "ais_canary_source": payload.defenses.canary_source,
            "ais_output_scanning": payload.defenses.output_scanning,
            "ais_tool_scanning": payload.defenses.tool_scanning,
            "ais_nimbus_lite": payload.defenses.nimbus_lite,
        },
    }
    route = "/chat" if payload.route == "/chat" else "/v1/responses"
    return run_responses_pipeline(
        db,
        body=body,
        headers=dict(request.headers),
        route=route,
        user_input=payload.user_input,
    )
