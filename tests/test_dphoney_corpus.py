"""U1: synthetic reference corpus + format specs.

Pure-data tests — no sklearn/torch needed, so they run under the default
`uv run pytest`.
"""

from __future__ import annotations

import pathlib

import dphoney.corpus as corpus_mod
from dphoney.corpus import FORMAT_NAMES, FORMATS, build_reference_corpus


def test_every_reference_string_is_format_valid():
    corpus = build_reference_corpus(seed=1, n_per_format=20)
    for name, values in corpus.by_format.items():
        spec = FORMATS[name]
        assert len(values) == 20
        for value in values:
            assert spec.validate(value), f"{name}: {value!r} failed its own spec"


def test_all_six_formats_present():
    corpus = build_reference_corpus(seed=1, n_per_format=5)
    assert set(corpus.by_format) == set(FORMAT_NAMES)
    assert len(FORMAT_NAMES) == 6


def test_deterministic_for_fixed_seed():
    a = build_reference_corpus(seed=7, n_per_format=10)
    b = build_reference_corpus(seed=7, n_per_format=10)
    assert a.by_format == b.by_format


def test_varies_with_seed():
    a = build_reference_corpus(seed=1, n_per_format=10)
    b = build_reference_corpus(seed=2, n_per_format=10)
    assert a.by_format != b.by_format


def test_reference_structure_is_non_uniform():
    # The whole contrast depends on the reference NOT being uniform-random: a
    # peaky Dirichlet(0.3) profile must put mass well above the uniform rate.
    corpus = build_reference_corpus(seed=3, n_per_format=5)
    for profile in corpus.profiles.values():
        uniform = 1.0 / profile.shape[1]
        assert profile.max() > uniform * 3


def test_module_stays_light_and_app_free():
    # R6 / KTD: the corpus is synthetic and importable from the base install —
    # no app coupling, no torch/sklearn at module level. Check import lines only
    # (the docstring legitimately mentions torch/sklearn in prose).
    src = pathlib.Path(corpus_mod.__file__).read_text()
    imports = "\n".join(ln for ln in src.splitlines() if ln.startswith(("import ", "from ")))
    assert "app" not in imports
    assert "torch" not in imports and "sklearn" not in imports


def test_validate_rejects_malformed():
    spec = FORMATS["github_pat"]
    assert spec.validate("ghp_" + "a" * 36)
    assert not spec.validate("ghp_" + "a" * 35)  # body too short
    assert not spec.validate("xxx_" + "a" * 36)  # wrong prefix
    assert not spec.validate("ghp_" + "a" * 35 + "!")  # bad body char


def test_sensitivity_matches_body_transitions():
    # ℓ1 sensitivity is the within-segment transition count, not 1.
    assert FORMATS["github_pat"].n_bigram_transitions == 35  # 36-char body
    # jwt_like has three body segments (18, 24, 22): (17 + 23 + 21).
    assert FORMATS["jwt_like"].n_bigram_transitions == 61
