from __future__ import annotations

from collections.abc import Iterable
from typing import Any


def flatten_json_strings(value: Any, prefix: str = "arguments") -> Iterable[tuple[str, str]]:
    if isinstance(value, dict):
        for key, child in value.items():
            yield from flatten_json_strings(child, f"{prefix}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from flatten_json_strings(child, f"{prefix}[{index}]")
    elif isinstance(value, str):
        yield prefix, value
