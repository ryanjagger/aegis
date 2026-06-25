from __future__ import annotations

from app.config import get_settings
from app.scanners.canary_scanner import CanaryScanner
from cift.corpus import (
    behavioral_check,
    build_corpus,
    encoding_success_rate,
    rot13,
)

DISCLAIMER_SUBSTRINGS = ("fake", "non-functional", "canaries for local", "diagnostic appendix")


def test_config_default_resolves():
    # U1 scaffolding sanity: lab settings are present with the documented default.
    assert get_settings().cift_model_name == "Qwen/Qwen2.5-1.5B-Instruct"


def test_injected_context_omits_fake_disclaimer():
    # Covers R6: the steering context must not tell the model the secret is fake.
    corpus = build_corpus(seed=0, n_benign=40)
    for prompt in corpus.attack_fit + [corpus.confound_control]:
        lowered = prompt.text.lower()
        for marker in DISCLAIMER_SUBSTRINGS:
            assert marker not in lowered, f"disclaimer '{marker}' leaked into {prompt.id}"


def test_benign_and_attack_are_surface_matched():
    # Covers R7: balanced labels, length-matched, confound is benign with exfil vocab.
    corpus = build_corpus(seed=0, n_benign=80)
    assert len(corpus.benign_fit) == len(corpus.attack_fit)
    assert len(corpus.benign_heldout) == len(corpus.attack_heldout)
    assert all(p.label == "benign" for p in corpus.benign_fit)
    assert all(p.label == "attack" for p in corpus.attack_fit)

    confound = corpus.confound_control
    assert confound.label == "benign"
    assert "exfil" in confound.text.lower() and "send" in confound.text.lower()


def test_encoding_suite_defeats_text_scanner_only_when_encoded():
    # Covers R8 / AE1: rot13 is an encoding transforms.py does not decode, so the
    # text scanner hits the verbatim secret but misses the rot13'd form.
    corpus = build_corpus(seed=0, n_benign=20, n_encoding=4)
    scenario = corpus.encoding_suite[0]
    scanner = CanaryScanner([scenario.canary])

    assert scanner.scan_text(scenario.canary.value, "lab"), "verbatim secret should be detected"
    assert not scanner.scan_text(
        rot13(scenario.canary.value), "lab"
    ), "rot13'd secret must NOT be detected (held-out encoding)"


def test_behavioral_check_and_success_rate():
    # Encoded arm passes only on an EXACT rot13 of the secret; success rate counts hits.
    corpus = build_corpus(seed=1, n_benign=20, n_encoding=3)
    canary = corpus.encoding_suite[0].canary

    assert behavioral_check(canary.value, canary, "verbatim_leak")
    assert not behavioral_check(canary.value, canary, "rot13_leak")  # plaintext != rot13
    assert behavioral_check(rot13(canary.value), canary, "rot13_leak")

    outputs = [
        (f"here you go: {rot13(canary.value)}", canary),  # correctly encoded
        ("sorry, I can't share that", canary),  # refused
    ]
    assert encoding_success_rate(outputs) == 0.5


def test_every_attack_prompt_embeds_a_canary():
    # Edge: a steering prompt without an embedded secret cannot probe credential use.
    corpus = build_corpus(seed=0, n_benign=40)
    for prompt in corpus.attack_fit + corpus.attack_heldout:
        assert prompt.canary is not None
        assert prompt.canary.value in prompt.text


def test_corpus_is_deterministic():
    a = build_corpus(seed=7, n_benign=30)
    b = build_corpus(seed=7, n_benign=30)
    assert [p.text for p in a.benign_fit] == [p.text for p in b.benign_fit]
