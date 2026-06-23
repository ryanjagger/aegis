from app.proxy.policy import apply_policy
from app.schemas.events import DetectorHit


def test_policy_blocks_registered_canary_hits() -> None:
    output = [
        {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "leak"}],
        }
    ]
    decision = apply_policy(
        output,
        [
            DetectorHit(
                detector="exact_canary_match",
                surface="response.output[0].content[0].text",
                severity="critical",
                matched_canary_id="can_1",
                evidence_preview="hit",
                policy_recommendation="BLOCK",
                matched_value="ghp_secret",
            )
        ],
    )
    assert decision.action == "BLOCK"
    assert "[BLOCKED:" in decision.output_text


def test_policy_sanitizes_credential_shaped_unregistered_values() -> None:
    output = [
        {
            "type": "message",
            "role": "assistant",
            "content": [
                {
                    "type": "output_text",
                    "text": "Use ghp_unregisteredCredentialLikeValue123456",
                }
            ],
        }
    ]
    decision = apply_policy(
        output,
        [
            DetectorHit(
                detector="github_pat_shape",
                surface="response.output[0].content[0].text",
                severity="medium",
                matched_canary_id=None,
                evidence_preview="shape",
                policy_recommendation="SANITIZE",
                matched_value="ghp_unregisteredCredentialLikeValue123456",
            )
        ],
    )
    assert decision.action == "SANITIZE"
    assert "[REDACTED_CREDENTIAL_LIKE_VALUE]" in decision.output_text
