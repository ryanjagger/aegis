def test_post_chat_uses_same_pipeline(client) -> None:
    response = client.post(
        "/chat",
        json={
            "session_id": "chat_session",
            "user_input": "Summarize this ticket.",
            "scenario": "direct_leak",
            "defenses": {
                "canary_injection": True,
                "output_scanning": True,
                "tool_scanning": True,
                "nimbus_lite": False,
            },
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["metadata"]["ais_policy_action"] == "BLOCK"
    assert body["metadata"]["ais_request_id"].startswith("req_")
