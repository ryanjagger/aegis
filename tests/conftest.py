from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.db.database import SessionLocal, init_db
from app.db.repository import clear_all
from app.main import app


@pytest.fixture(autouse=True)
def clean_db() -> None:
    init_db()
    with SessionLocal() as db:
        clear_all(db)
    yield
    with SessionLocal() as db:
        clear_all(db)


@pytest.fixture
def client() -> TestClient:
    with TestClient(app) as test_client:
        yield test_client
