from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Request
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db.database import get_db
from app.proxy.responses_proxy import run_responses_pipeline
from app.schemas.api import ChatRequest

router = APIRouter()


@router.post("/chat")
def chat(
    request: Request,
    payload: ChatRequest,
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
            "ais_output_scanning": payload.defenses.output_scanning,
            "ais_tool_scanning": payload.defenses.tool_scanning,
            "ais_nimbus_lite": payload.defenses.nimbus_lite,
        },
    }
    return run_responses_pipeline(
        db,
        body=body,
        headers=dict(request.headers),
        route="/chat",
        user_input=payload.user_input,
    )
