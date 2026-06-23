from __future__ import annotations

import json
from typing import Any

from app.scanners.canary_scanner import CanaryScanner
from app.scanners.credential_shapes import CredentialShapeDetector
from app.schemas.events import DetectorHit
from app.tools.argument_flattener import flatten_json_strings


class ResponsesItemScanner:
    def __init__(self, canary_scanner: CanaryScanner) -> None:
        self.canary_scanner = canary_scanner
        self.credential_detector = CredentialShapeDetector(canary_scanner.registered_values)

    def _scan_text(self, text: str, surface: str) -> list[DetectorHit]:
        return [
            *self.canary_scanner.scan_text(text, surface),
            *self.credential_detector.scan_text(text, surface),
        ]

    def scan_output(
        self,
        output: list[dict[str, Any]],
        *,
        scan_messages: bool = True,
        scan_tool_items: bool = True,
        scan_serialized_output: bool = True,
    ) -> list[DetectorHit]:
        hits: list[DetectorHit] = []
        for index, item in enumerate(output):
            item_type = item.get("type")
            base = f"response.output[{index}]"
            if item_type == "message" and scan_messages:
                hits.extend(self._scan_message_item(item, base))
            elif item_type == "function_call" and scan_tool_items:
                hits.extend(self._scan_function_call_item(item, base))
            elif item_type == "function_call_output" and scan_tool_items:
                output_value = item.get("output")
                if isinstance(output_value, str):
                    hits.extend(self._scan_text(output_value, f"{base}.output"))
                else:
                    hits.extend(self._scan_text(json.dumps(output_value), f"{base}.output"))

        if scan_serialized_output:
            serialized = json.dumps(output, sort_keys=True)
            hits.extend(self._scan_text(serialized, "response.output_json"))
        return _dedupe_hits(hits)

    def _scan_message_item(self, item: dict[str, Any], base: str) -> list[DetectorHit]:
        content = item.get("content")
        hits: list[DetectorHit] = []
        if isinstance(content, str):
            hits.extend(self._scan_text(content, f"{base}.content"))
        elif isinstance(content, list):
            for part_index, part in enumerate(content):
                if isinstance(part, dict):
                    text = part.get("text")
                    if isinstance(text, str):
                        hits.extend(self._scan_text(text, f"{base}.content[{part_index}].text"))
                elif isinstance(part, str):
                    hits.extend(self._scan_text(part, f"{base}.content[{part_index}]"))
        return hits

    def _scan_function_call_item(self, item: dict[str, Any], base: str) -> list[DetectorHit]:
        arguments = item.get("arguments", "")
        hits: list[DetectorHit] = []
        if isinstance(arguments, str):
            hits.extend(self._scan_text(arguments, f"{base}.function_call.arguments"))
            try:
                parsed = json.loads(arguments)
            except json.JSONDecodeError:
                parsed = None
        else:
            parsed = arguments
            hits.extend(self._scan_text(json.dumps(arguments), f"{base}.function_call.arguments"))

        if parsed is not None:
            for surface, value in flatten_json_strings(parsed, f"{base}.function_call.arguments"):
                hits.extend(self._scan_text(value, surface))
            hits.extend(
                self._scan_text(
                    json.dumps(parsed, sort_keys=True),
                    f"{base}.function_call.arguments_json",
                )
            )
        return hits


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
