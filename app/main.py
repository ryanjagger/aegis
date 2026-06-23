from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api import chat, dashboard_data, playground, responses
from app.config import get_settings
from app.db.database import init_db

settings = get_settings()


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    init_db()
    yield


app = FastAPI(title=settings.app_name, lifespan=lifespan)


@app.get("/")
def root() -> dict[str, str]:
    return {"name": settings.app_name, "status": "ok"}


app.include_router(responses.router)
app.include_router(chat.router)
app.include_router(dashboard_data.router)
app.include_router(playground.router)
