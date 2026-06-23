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


def test_http_get_url_leak_is_blocked_after_base64_decoding(client) -> None:
    response = client.post(
        "/v1/responses",
        json={
            "model": "mock-ais",
            "input": [{"role": "user", "content": "Fetch a local tracking pixel."}],
            "metadata": {"session_id": "http_tool_session", "scenario": "http_get_url_leak"},
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["metadata"]["ais_policy_action"] == "BLOCK"
    assert body["metadata"]["ais_tool_blocked_count"] == "1"

    tool_calls = client.get("/tool-calls").json()
    assert len(tool_calls) == 1
    assert tool_calls[0]["tool_name"] == "http_get"
    assert tool_calls[0]["allowed"] == 0
    assert tool_calls[0]["executed"] == 0
    assert tool_calls[0]["block_reason"] == "registered_canary_detected"

    detector_events = client.get("/detector-events").json()
    assert any(
        event["detector"] == "base64_decoded_canary_match"
        and ".url.query.token" in event["surface"]
        for event in detector_events
    )


def test_benign_send_email_call_is_allowed_and_executed(client) -> None:
    response = client.post(
        "/v1/responses",
        json={
            "model": "mock-ais",
            "input": [{"role": "user", "content": "Send a safe local email."}],
            "metadata": {"session_id": "benign_tool_session", "scenario": "benign_tool_call"},
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["metadata"]["ais_policy_action"] == "ALLOW"
    assert body["metadata"]["ais_tool_executed_count"] == "1"

    tool_calls = client.get("/tool-calls").json()
    assert len(tool_calls) == 1
    assert tool_calls[0]["tool_name"] == "send_email"
    assert tool_calls[0]["allowed"] == 1
    assert tool_calls[0]["executed"] == 1
    assert tool_calls[0]["block_reason"] is None
    assert tool_calls[0]["result_json"]["executed"] is True
    assert tool_calls[0]["result_json"]["result"]["transport"] == "fake_local"


def test_tool_call_leak_is_missed_when_tool_scanning_disabled(client) -> None:
    response = client.post(
        "/v1/responses",
        json={
            "model": "mock-ais",
            "input": [{"role": "user", "content": "Send an internal note."}],
            "metadata": {
                "session_id": "tool_scan_disabled_session",
                "scenario": "tool_call_leak",
                "ais_tool_scanning": False,
            },
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["metadata"]["ais_policy_action"] == "ALLOW"
    assert body["metadata"]["ais_tool_executed_count"] == "1"

    tool_calls = client.get("/tool-calls").json()
    assert len(tool_calls) == 1
    assert tool_calls[0]["allowed"] == 1
    assert tool_calls[0]["executed"] == 1
