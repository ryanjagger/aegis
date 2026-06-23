from __future__ import annotations

from typing import Any


def send_email(to: str, subject: str, body: str) -> dict[str, Any]:
    return {"sent": True, "to": to, "subject": subject, "body_preview": body[:120]}


def http_get(url: str, headers: dict[str, str] | None = None) -> dict[str, Any]:
    return {"status_code": 200, "url": url, "headers": headers or {}, "body": "fake local body"}


def create_ticket(title: str, description: str) -> dict[str, Any]:
    return {"ticket_id": "ticket_local_001", "title": title, "description": description}


def write_file(path: str, content: str) -> dict[str, Any]:
    return {"path": path, "bytes": len(content), "written": False}


def query_kb(query: str) -> dict[str, Any]:
    return {"query": query, "results": ["fake local knowledge-base result"]}
