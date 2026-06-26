"""U2: DP bigram generator — Laplace ε-DP, format-valid sampling, persistence."""

from __future__ import annotations

import numpy as np
import pytest

from dphoney import generator
from dphoney.corpus import FORMAT_NAMES, FORMATS, build_reference_corpus
from dphoney.generator import (
    add_laplace_noise,
    fit_bigram_counts,
    fit_dp_model,
    laplace_scale,
    load_dp_model,
    sample_canary,
    save_dp_model,
    to_transition_probs,
)


@pytest.fixture(scope="module")
def models():
    reference = build_reference_corpus(seed=5, n_per_format=80)
    return fit_dp_model(reference, epsilon=1.0, seed=0)


def test_sampled_canaries_are_format_valid(models):
    rng = np.random.default_rng(0)
    for name, model in models.items():
        spec = FORMATS[name]
        for _ in range(25):
            assert spec.validate(sample_canary(model, rng))


def test_structured_format_scaffolding_is_templated(models):
    rng = np.random.default_rng(1)
    value = sample_canary(models["postgres_url"], rng)
    assert value.startswith("postgres://ais_user:")
    assert "@db.local:5432/ais_demo_" in value
    assert FORMATS["postgres_url"].validate(value)


def test_sampling_exhaustion_falls_back(monkeypatch, models):
    # Force every modelled sample to be invalid (empty body) so the retry cap is
    # exhausted and the defensive fallback fires, still returning a valid value.
    monkeypatch.setattr(generator, "_sample_body", lambda *a, **k: "")
    rng = np.random.default_rng(2)
    value = sample_canary(models["github_pat"], rng, max_retries=2)
    assert FORMATS["github_pat"].validate(value)


def test_smaller_epsilon_means_larger_scale():
    s = 35.0
    assert laplace_scale(s, 0.1) > laplace_scale(s, 1.0) > laplace_scale(s, 10.0)


def test_smaller_epsilon_deviates_more_from_counts():
    spec = FORMATS["github_pat"]
    reference = build_reference_corpus(seed=2, n_per_format=60)
    counts = fit_bigram_counts(spec, reference.by_format["github_pat"])
    s = spec.n_bigram_transitions
    low_noise = add_laplace_noise(counts, 1000.0, s, np.random.default_rng(0))
    high_noise = add_laplace_noise(counts, 0.1, s, np.random.default_rng(0))
    assert np.abs(low_noise - counts).mean() < np.abs(high_noise - counts).mean()


def test_sensitivity_is_bigram_transitions_not_one(models):
    for name, model in models.items():
        assert model.sensitivity == FORMATS[name].n_bigram_transitions
        assert model.sensitivity > 1  # the bug we fixed: it is not 1.0


def test_zero_row_smoothed_to_uniform():
    counts = np.array([[0.0, 0.0, 0.0], [2.0, 1.0, 1.0]])
    probs = to_transition_probs(counts)
    np.testing.assert_allclose(probs[0], [1 / 3, 1 / 3, 1 / 3])
    np.testing.assert_allclose(probs.sum(axis=1), [1.0, 1.0])


def test_seeded_fit_is_reproducible():
    reference = build_reference_corpus(seed=1, n_per_format=40)
    a = fit_dp_model(reference, epsilon=1.0, seed=0)
    b = fit_dp_model(reference, epsilon=1.0, seed=0)
    for name in FORMAT_NAMES:
        np.testing.assert_array_equal(a[name].trans_probs, b[name].trans_probs)
    c = fit_dp_model(reference, epsilon=1.0, seed=99)
    assert not np.array_equal(a["github_pat"].trans_probs, c["github_pat"].trans_probs)


def test_model_round_trips(tmp_path, models):
    path = tmp_path / "bigram_model.npz"
    save_dp_model(models, path)
    loaded = load_dp_model(path)
    assert set(loaded) == set(models)
    for name, model in models.items():
        np.testing.assert_array_equal(loaded[name].trans_probs, model.trans_probs)
        assert loaded[name].sensitivity == model.sensitivity
        assert loaded[name].epsilon == model.epsilon
