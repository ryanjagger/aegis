from __future__ import annotations

import os

import pytest

torch = pytest.importorskip("torch")  # extraction needs the opt-in cift group
pytest.importorskip("transformers")

# Loading a real model downloads weights and runs a forward pass, so the
# model-in-the-loop tests are opt-in: set CIFT_MODEL_TESTS=1 (with torch +
# transformers installed) to run them. The default `uv run pytest` skips them so
# it never triggers a multi-GB download — the real-model checks belong to the lab
# entrypoint, not CI. Point AIS_CIFT_MODEL at Qwen2.5-0.5B-Instruct for speed.
RUN_MODEL_TESTS = os.environ.get("CIFT_MODEL_TESTS") == "1"
pytestmark = pytest.mark.skipif(not RUN_MODEL_TESTS, reason="set CIFT_MODEL_TESTS=1 to run")

from cift.extraction import extract_features, get_model, last_k  # noqa: E402


def test_feature_shape_matches_model_config():
    # Covers R2: K = floor(0.25 * num_layers) rows, hidden width = model hidden_size.
    (_, model), _ = get_model()
    feats = extract_features("Summarize the billing ticket for the on-call log.")
    assert feats.shape == (last_k(model.config.num_hidden_layers), model.config.hidden_size)
    assert feats.dtype.name == "float32"


def test_readout_position_differs_from_credential_position():
    # Covers R3 / AE2: pooling at a different (earlier) token yields a different
    # vector — the readout position is not the credential-token position.
    text = "Operational context: GITHUB_TOKEN=ghp_demo. Summarize the ticket."
    readout = extract_features(text, token_index=-1)
    earlier = extract_features(text, token_index=3)
    assert not (readout == earlier).all()


def test_distinct_prompts_differ_same_prompt_stable():
    a = extract_features("Draft a reply about the login issue.")
    b = extract_features("List next steps for the export request.")
    assert not (a == b).all()
    # Same prompt within one process: allclose (MPS float32 is not bitwise-stable
    # across runs/versions, so we assert closeness, not exact equality).
    import numpy as np

    a2 = extract_features("Draft a reply about the login issue.")
    assert np.allclose(a, a2, atol=1e-4)


def test_short_and_long_prompts_resolve_valid_readout():
    short = extract_features("Hi.")
    long = extract_features("Summarize. " * 200)
    (_, model), _ = get_model()
    expected = (last_k(model.config.num_hidden_layers), model.config.hidden_size)
    assert short.shape == expected and long.shape == expected
