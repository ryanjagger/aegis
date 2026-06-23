from __future__ import annotations

import base64
import binascii
import json
import re
from collections.abc import Iterable
from urllib.parse import unquote_plus

BASE64_CANDIDATE_RE = re.compile(r"\b[A-Za-z0-9+/_-]{16,}={0,2}\b")
HEX_CANDIDATE_RE = re.compile(r"\b(?:[0-9a-fA-F]{2}){8,}\b")
MARKDOWN_URL_RE = re.compile(r"\[[^\]]+\]\(([^)\s]+)\)")


def _decode_base64_candidate(value: str) -> str | None:
    candidate = value.strip()
    if len(candidate) < 16:
        return None
    normalized = candidate.replace("-", "+").replace("_", "/")
    padding = (-len(normalized)) % 4
    normalized += "=" * padding
    try:
        decoded = base64.b64decode(normalized.encode("ascii"), validate=False)
        text = decoded.decode("utf-8")
    except (binascii.Error, UnicodeDecodeError, ValueError):
        return None
    if not text or sum(ch.isprintable() for ch in text) / max(len(text), 1) < 0.85:
        return None
    return text


def _decode_hex_candidate(value: str) -> str | None:
    try:
        decoded = bytes.fromhex(value).decode("utf-8")
    except (ValueError, UnicodeDecodeError):
        return None
    if not decoded or sum(ch.isprintable() for ch in decoded) / max(len(decoded), 1) < 0.85:
        return None
    return decoded


def _json_unescape(value: str) -> str | None:
    if "\\" not in value:
        return None
    try:
        return json.loads(f'"{value}"')
    except json.JSONDecodeError:
        try:
            return value.encode("utf-8").decode("unicode_escape")
        except UnicodeDecodeError:
            return None


def transformed_texts(text: str) -> Iterable[tuple[str, str]]:
    yield "exact_canary_match", text
    yield "case_normalized_canary_match", text.lower()
    yield "whitespace_stripped_canary_match", re.sub(r"\s+", "", text)

    url_decoded = unquote_plus(text)
    if url_decoded != text or "://" in text or "%" in text or "+" in text:
        yield "url_decoded_canary_match", url_decoded

    json_unescaped = _json_unescape(text)
    if json_unescaped and json_unescaped != text:
        yield "json_unescaped_canary_match", json_unescaped

    for url in MARKDOWN_URL_RE.findall(text):
        yield "markdown_url_canary_match", url
        decoded_url = unquote_plus(url)
        if decoded_url != url:
            yield "url_decoded_canary_match", decoded_url

    for candidate in BASE64_CANDIDATE_RE.findall(text):
        decoded = _decode_base64_candidate(candidate)
        if decoded:
            yield "base64_decoded_canary_match", decoded

    whole_base64 = _decode_base64_candidate(text)
    if whole_base64:
        yield "base64_decoded_canary_match", whole_base64

    for candidate in HEX_CANDIDATE_RE.findall(text):
        decoded = _decode_hex_candidate(candidate)
        if decoded:
            yield "hex_decoded_canary_match", decoded

    stripped = text.strip()
    if HEX_CANDIDATE_RE.fullmatch(stripped):
        decoded = _decode_hex_candidate(stripped)
        if decoded:
            yield "hex_decoded_canary_match", decoded
