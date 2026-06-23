import json

from app.scanners.canary_scanner import CanaryScanner
from app.scanners.responses_item_scanner import ResponsesItemScanner

CANARY = {
    "id": "can_test",
    "value": "sk_live_testValueForResponsesScanner123",
}


def test_responses_item_scanner_scans_message_items() -> None:
    output = [
        {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": f"token={CANARY['value']}"}],
        }
    ]
    hits = ResponsesItemScanner(CanaryScanner([CANARY])).scan_output(output)
    assert any(hit.surface == "response.output[0].content[0].text" for hit in hits)


def test_responses_item_scanner_scans_function_call_arguments() -> None:
    output = [
        {
            "type": "function_call",
            "call_id": "call_1",
            "name": "send_email",
            "arguments": json.dumps({"body": f"token={CANARY['value']}"}),
        }
    ]
    hits = ResponsesItemScanner(CanaryScanner([CANARY])).scan_output(output)
    assert any("function_call.arguments.body" in hit.surface for hit in hits)
