from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass
from enum import IntEnum
from typing import Any

from app.schemas.events import DetectorHit


class PolicyRank(IntEnum):
    ALLOW = 0
    WARN = 1
    SANITIZE = 2
    BLOCK = 3
    QUARANTINE = 4


@dataclass(frozen=True)
class PolicyDecision:
    action: str
    output: list[dict[str, Any]]
    output_text: str


def _output_text(output: list[dict[str, Any]]) -> str:
    chunks: list[str] = []
    for item in output:
        if item.get("type") != "message":
            continue
        content = item.get("content")
        if isinstance(content, str):
            chunks.append(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    chunks.append(part["text"])
                elif isinstance(part, str):
                    chunks.append(part)
    return "\n".join(chunks)


def _rank_for_hit(hit: DetectorHit) -> PolicyRank:
    if hit.policy_recommendation == "BLOCK":
        return PolicyRank.BLOCK
    if hit.policy_recommendation == "SANITIZE":
        return PolicyRank.SANITIZE
    if hit.policy_recommendation == "WARN":
        return PolicyRank.WARN
    return PolicyRank.ALLOW


def _rank_for_action(action: str | None) -> PolicyRank:
    if not action:
        return PolicyRank.ALLOW
    try:
        return PolicyRank[action]
    except KeyError:
        return PolicyRank.ALLOW


def apply_policy(
    output: list[dict[str, Any]],
    hits: list[DetectorHit],
    *,
    nimbus_action: str | None = None,
) -> PolicyDecision:
    nimbus_rank = _rank_for_action(nimbus_action)
    if not hits and nimbus_rank == PolicyRank.ALLOW:
        text = _output_text(output)
        return PolicyDecision(action="ALLOW", output=output, output_text=text)

    hit_rank = max((_rank_for_hit(hit) for hit in hits), default=PolicyRank.ALLOW)
    rank = max(hit_rank, nimbus_rank)
    action = PolicyRank(rank).name

    if rank >= PolicyRank.BLOCK:
        blocked_text = (
            "[BLOCKED: registered canary detected in model output]"
            if hit_rank >= PolicyRank.BLOCK
            else "[BLOCKED: cumulative leakage budget exceeded]"
        )
        blocked_output = [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": blocked_text}],
            }
        ]
        return PolicyDecision(action=action, output=blocked_output, output_text=blocked_text)

    if rank == PolicyRank.SANITIZE:
        sanitized = sanitize_output(output, hits)
        return PolicyDecision(action=action, output=sanitized, output_text=_output_text(sanitized))

    return PolicyDecision(action=action, output=output, output_text=_output_text(output))


def sanitize_output(output: list[dict[str, Any]], hits: list[DetectorHit]) -> list[dict[str, Any]]:
    sanitized = copy.deepcopy(output)
    replacements: list[tuple[str, str]] = []
    for hit in hits:
        if not hit.matched_value:
            continue
        replacement = (
            "[REDACTED_CANARY]"
            if hit.matched_canary_id
            else "[REDACTED_CREDENTIAL_LIKE_VALUE]"
        )
        replacements.append((hit.matched_value, replacement))
    if not replacements:
        return sanitized
    return _replace_strings(sanitized, replacements)


def _replace_strings(value: Any, replacements: list[tuple[str, str]]) -> Any:
    if isinstance(value, dict):
        return {key: _replace_strings(child, replacements) for key, child in value.items()}
    if isinstance(value, list):
        return [_replace_strings(child, replacements) for child in value]
    if isinstance(value, str):
        text = value
        for matched, replacement in replacements:
            text = text.replace(matched, replacement)
            text = re.sub(re.escape(matched), replacement, text, flags=re.IGNORECASE)
            try:
                json_matched = json.dumps(matched)[1:-1]
                text = text.replace(json_matched, replacement)
            except TypeError:
                pass
        return text
    return value
