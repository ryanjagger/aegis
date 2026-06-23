def _responses_payload(scenario: str) -> dict:
    return {
        "model": "mock-ais",
        "input": [{"role": "user", "content": "Summarize this support ticket."}],
        "store": False,
        "metadata": {"session_id": f"sess_{scenario}", "scenario": scenario},
    }


def test_post_responses_benign_returns_allow(client) -> None:
    response = client.post("/v1/responses", json=_responses_payload("benign"))
    assert response.status_code == 200
    body = response.json()
    assert body["metadata"]["ais_policy_action"] == "ALLOW"


def test_post_responses_direct_leak_returns_block(client) -> None:
    response = client.post("/v1/responses", json=_responses_payload("direct_leak"))
    assert response.status_code == 200
    body = response.json()
    assert body["metadata"]["ais_policy_action"] == "BLOCK"
    assert "[BLOCKED:" in body["output"][0]["content"][0]["text"]


def test_post_responses_base64_leak_returns_block(client) -> None:
    response = client.post("/v1/responses", json=_responses_payload("base64_leak"))
    assert response.status_code == 200
    body = response.json()
    assert body["metadata"]["ais_policy_action"] == "BLOCK"


def test_post_responses_hex_and_markdown_leaks_return_block(client) -> None:
    for scenario in ("hex_leak", "markdown_link_leak"):
        response = client.post("/v1/responses", json=_responses_payload(scenario))
        assert response.status_code == 200
        assert response.json()["metadata"]["ais_policy_action"] == "BLOCK"
