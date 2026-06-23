def test_tool_call_leak_is_blocked_and_recorded(client) -> None:
    response = client.post(
        "/v1/responses",
        json={
            "model": "mock-ais",
            "input": [{"role": "user", "content": "Send an internal note."}],
            "metadata": {"session_id": "tool_session", "scenario": "tool_call_leak"},
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["metadata"]["ais_policy_action"] == "BLOCK"

    tool_calls = client.get("/tool-calls").json()
    assert len(tool_calls) == 1
    assert tool_calls[0]["tool_name"] == "send_email"
    assert tool_calls[0]["allowed"] == 0
    assert tool_calls[0]["executed"] == 0
    assert tool_calls[0]["block_reason"] == "registered_canary_detected"
