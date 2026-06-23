from __future__ import annotations

from collections.abc import Callable
from typing import Any


def send_email(to: str, subject: str, body: str) -> dict[str, Any]:
    return {
        "sent": True,
        "transport": "fake_local",
        "to": to,
        "subject": subject,
        "body_preview": body[:120],
    }


def http_get(url: str, headers: dict[str, str] | None = None) -> dict[str, Any]:
    return {
        "status_code": 200,
        "transport": "fake_local",
        "url": url,
        "headers": headers or {},
        "body": "fake local body",
    }


def create_ticket(title: str, description: str) -> dict[str, Any]:
    return {"ticket_id": "ticket_local_001", "title": title, "description": description}


def write_file(path: str, content: str) -> dict[str, Any]:
    return {"path": path, "bytes": len(content), "written": False}


def query_kb(query: str) -> dict[str, Any]:
    return {"query": query, "results": ["fake local knowledge-base result"]}


FAKE_TOOLS: dict[str, Callable[..., dict[str, Any]]] = {
    "send_email": send_email,
    "http_get": http_get,
    "create_ticket": create_ticket,
    "write_file": write_file,
    "query_kb": query_kb,
}
