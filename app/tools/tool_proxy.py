from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from urllib.parse import parse_qsl, unquote_plus, urlparse

from sqlalchemy.orm import Session

from app.db import repository
from app.scanners.canary_scanner import CanaryScanner
from app.scanners.credential_shapes import CredentialShapeDetector
from app.schemas.events import DetectorHit
from app.tools.argument_flattener import flatten_json_strings
from app.tools.fake_tools import FAKE_TOOLS


@dataclass(frozen=True)
class ToolProxyResult:
    call_id: str
    tool_name: str
    arguments: Any
    allowed: bool
    executed: bool
    result: Any
    block_reason: str | None
    hits: list[DetectorHit]


class ToolProxy:
    def __init__(
        self,
        db: Session,
        *,
        request_id: str,
        response_id: str,
        session_id: str,
        canary_scanner: CanaryScanner,
        scan_enabled: bool,
    ) -> None:
        self.db = db
        self.request_id = request_id
        self.response_id = response_id
        self.session_id = session_id
        self.canary_scanner = canary_scanner
        self.credential_detector = CredentialShapeDetector(canary_scanner.registered_values)
        self.scan_enabled = scan_enabled

    def handle_function_call(self, item: dict[str, Any], *, item_index: int) -> ToolProxyResult:
        call_id = str(item.get("call_id") or repository.short_id("call"))
        tool_name = str(item.get("name") or "unknown")
        arguments = self._parse_arguments(item.get("arguments"))
        surface_base = f"response.output[{item_index}].function_call.arguments"

        hits: list[DetectorHit] = []
        if self.scan_enabled:
            hits = self._scan_arguments(arguments, surface_base)
            self._persist_hits(hits)

        blocked_by_hits = any(
            hit.policy_recommendation in {"BLOCK", "SANITIZE", "QUARANTINE"} for hit in hits
        )
        known_tool = tool_name in FAKE_TOOLS
        allowed = self.scan_enabled and not blocked_by_hits and known_tool
        executed = False
        result: Any = None
        block_reason: str | None = None

        if not self.scan_enabled:
            allowed = True
            result = self._execute_fake_tool(tool_name, arguments)
            executed = result.get("executed", False) if isinstance(result, dict) else False
            block_reason = "tool_scanning_disabled"
        elif blocked_by_hits:
            block_reason = "registered_canary_detected"
        elif not known_tool:
            block_reason = "unknown_tool"
        else:
            result = self._execute_fake_tool(tool_name, arguments)
            executed = result.get("executed", False) if isinstance(result, dict) else False
            if not executed:
                allowed = False
                block_reason = "tool_execution_failed"

        if self.scan_enabled and executed:
            result_hits = self._scan_result(result, f"tool_call.{call_id}.result")
            self._persist_hits(result_hits)
            hits = [*hits, *result_hits]
            if any(
                hit.policy_recommendation in {"BLOCK", "SANITIZE", "QUARANTINE"}
                for hit in result_hits
            ):
                allowed = False
                block_reason = "unsafe_tool_result_detected"

        repository.add_tool_call(
            self.db,
            request_id=self.request_id,
            response_id=self.response_id,
            session_id=self.session_id,
            call_id=call_id,
            tool_name=tool_name,
            arguments=arguments,
            allowed=allowed,
            executed=executed,
            result=result,
            block_reason=block_reason,
        )
        repository.add_event(
            self.db,
            request_id=self.request_id,
            response_id=self.response_id,
            session_id=self.session_id,
            event_type="tool_call.allowed" if allowed else "tool_call.blocked",
            payload={
                "tool_name": tool_name,
                "call_id": call_id,
                "allowed": allowed,
                "executed": executed,
                "block_reason": block_reason,
                "hit_count": len(hits),
            },
        )
        return ToolProxyResult(
            call_id=call_id,
            tool_name=tool_name,
            arguments=arguments,
            allowed=allowed,
            executed=executed,
            result=result,
            block_reason=block_reason,
            hits=hits,
        )

    def _parse_arguments(self, arguments: Any) -> Any:
        if isinstance(arguments, str):
            try:
                return json.loads(arguments)
            except json.JSONDecodeError:
                return arguments
        return arguments if arguments is not None else {}

    def _execute_fake_tool(self, tool_name: str, arguments: Any) -> dict[str, Any]:
        tool = FAKE_TOOLS.get(tool_name)
        if tool is None:
            return {"executed": False, "error": f"Unknown fake tool: {tool_name}"}
        if not isinstance(arguments, dict):
            return {"executed": False, "error": "Tool arguments must be a JSON object"}
        try:
            result = tool(**arguments)
        except TypeError as exc:
            return {"executed": False, "error": str(exc)}
        return {"executed": True, "result": result}

    def _scan_arguments(self, arguments: Any, surface_base: str) -> list[DetectorHit]:
        hits: list[DetectorHit] = []
        serialized = json.dumps(arguments, sort_keys=True, default=str)
        hits.extend(self._scan_text(serialized, f"{surface_base}_json"))

        if isinstance(arguments, str):
            hits.extend(self._scan_text(arguments, surface_base))
            hits.extend(self._scan_url(arguments, surface_base))
            return _dedupe_hits(hits)

        for surface, value in flatten_json_strings(arguments, surface_base):
            hits.extend(self._scan_text(value, surface))
            hits.extend(self._scan_url(value, surface))
        return _dedupe_hits(hits)

    def _scan_result(self, result: Any, surface_base: str) -> list[DetectorHit]:
        hits: list[DetectorHit] = []
        serialized = json.dumps(result, sort_keys=True, default=str)
        hits.extend(self._scan_text(serialized, f"{surface_base}_json"))
        for surface, value in flatten_json_strings(result, surface_base):
            hits.extend(self._scan_text(value, surface))
            hits.extend(self._scan_url(value, surface))
        return _dedupe_hits(hits)

    def _scan_text(self, text: str, surface: str) -> list[DetectorHit]:
        return [
            *self.canary_scanner.scan_text(text, surface),
            *self.credential_detector.scan_text(text, surface),
        ]

    def _scan_url(self, value: str, surface: str) -> list[DetectorHit]:
        parsed = urlparse(value)
        if not parsed.scheme or not parsed.netloc:
            return []

        hits: list[DetectorHit] = []
        decoded_url = unquote_plus(value)
        hits.extend(self._scan_text(decoded_url, f"{surface}.url_decoded"))
        if parsed.query:
            decoded_query = unquote_plus(parsed.query)
            hits.extend(self._scan_text(decoded_query, f"{surface}.url.query"))
            for key, query_value in parse_qsl(parsed.query, keep_blank_values=True):
                decoded_value = unquote_plus(query_value)
                hits.extend(self._scan_text(decoded_value, f"{surface}.url.query.{key}"))
        return hits

    def _persist_hits(self, hits: list[DetectorHit]) -> None:
        for hit in hits:
            repository.add_detector_hit(
                self.db,
                request_id=self.request_id,
                response_id=self.response_id,
                session_id=self.session_id,
                hit=hit,
            )
            if hit.matched_canary_id:
                repository.mark_canary_leaked(
                    self.db,
                    canary_id=hit.matched_canary_id,
                    response_id=self.response_id,
                    surface=hit.surface,
                )
            repository.add_event(
                self.db,
                request_id=self.request_id,
                response_id=self.response_id,
                session_id=self.session_id,
                event_type="detector.hit",
                payload={
                    "detector": hit.detector,
                    "surface": hit.surface,
                    "severity": hit.severity,
                    "matched_canary_id": hit.matched_canary_id,
                },
            )


def _dedupe_hits(hits: list[DetectorHit]) -> list[DetectorHit]:
    deduped: list[DetectorHit] = []
    seen: set[tuple[str, str, str | None, str | None]] = set()
    for hit in hits:
        key = (hit.detector, hit.surface, hit.matched_canary_id, hit.matched_value)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(hit)
    return deduped
