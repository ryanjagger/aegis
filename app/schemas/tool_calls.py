from typing import Any

from pydantic import BaseModel


class ToolCallTrace(BaseModel):
    call_id: str
    tool_name: str
    arguments: Any
    allowed: bool
    executed: bool = False
    block_reason: str | None = None
