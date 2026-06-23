from __future__ import annotations

from typing import Any


class ToolProxy:
    """Placeholder for Phase 2 fake tool dispatch.

    The updated MVP scans and records function_call Items, but it does not execute
    fake tools yet. Full argument scanning before fake execution belongs to Phase 2.
    """

    def execute(self, function_call: dict[str, Any]) -> dict[str, Any]:
        return {
            "executed": False,
            "reason": "Tool execution is intentionally deferred to Phase 2.",
            "function_call": function_call,
        }
