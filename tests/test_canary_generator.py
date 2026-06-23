from app.canaries.generator import CANARY_FORMATS, canary_hash, generate_canary


def test_canary_generation_prefixes() -> None:
    expected_prefixes = {
        "github_pat": "ghp_",
        "stripe_key": "sk_live_",
        "aws_access_key": "AKIA",
        "postgres_url": "postgres://",
        "support_token": "support_live_",
    }
    for format_name in CANARY_FORMATS:
        canary = generate_canary(format_name, "test_source")
        assert canary.id.startswith("can_")
        assert canary.value_hash == canary_hash(canary.value)
        if format_name == "jwt_like":
            assert len(canary.value.split(".")) == 3
        else:
            assert canary.value.startswith(expected_prefixes[format_name])
