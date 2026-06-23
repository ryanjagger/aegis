from __future__ import annotations

import base64
import hashlib
import secrets
import string
from dataclasses import dataclass

from app.db.repository import short_id

CANARY_FORMATS = (
    "github_pat",
    "stripe_key",
    "aws_access_key",
    "postgres_url",
    "jwt_like",
    "support_token",
)

_ALNUM = string.ascii_letters + string.digits
_UPPER_DIGITS = string.ascii_uppercase + string.digits
_B64URL = string.ascii_letters + string.digits + "-_"


@dataclass(frozen=True)
class GeneratedCanary:
    id: str
    value: str
    value_hash: str
    format: str
    source_label: str


def _token(length: int, alphabet: str = _ALNUM) -> str:
    return "".join(secrets.choice(alphabet) for _ in range(length))


def canary_hash(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def generate_canary(format_name: str, source_label: str) -> GeneratedCanary:
    if format_name == "github_pat":
        value = "ghp_" + _token(36)
    elif format_name == "stripe_key":
        value = "sk_live_" + _token(32)
    elif format_name == "aws_access_key":
        value = "AKIA" + _token(16, _UPPER_DIGITS)
    elif format_name == "postgres_url":
        value = f"postgres://ais_user:{_token(20)}@db.local:5432/ais_demo_{_token(6)}"
    elif format_name == "jwt_like":
        value = ".".join(_token(length, _B64URL) for length in (18, 24, 22))
    elif format_name == "support_token":
        value = "support_live_" + _token(28)
    else:
        raise ValueError(f"Unsupported canary format: {format_name}")

    return GeneratedCanary(
        id=short_id("can"),
        value=value,
        value_hash=canary_hash(value),
        format=format_name,
        source_label=source_label,
    )


def generate_canaries(
    *,
    formats: list[str] | tuple[str, ...] = CANARY_FORMATS[:5],
    source_labels: list[str] | tuple[str, ...],
) -> list[GeneratedCanary]:
    canaries: list[GeneratedCanary] = []
    for index, format_name in enumerate(formats):
        source_label = source_labels[index % len(source_labels)]
        canaries.append(generate_canary(format_name, source_label))
    return canaries


def first_canary_like_value(text: str) -> str | None:
    from app.scanners.credential_shapes import CREDENTIAL_PATTERNS

    for pattern in CREDENTIAL_PATTERNS.values():
        match = pattern.search(text)
        if match:
            return match.group(0)
    return None


def base64_encode(value: str) -> str:
    return base64.b64encode(value.encode("utf-8")).decode("ascii")
