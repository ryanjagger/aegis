"""U3: distinguisher battery + the two controls.

Needs the opt-in `dphoney` group (scikit-learn); skipped on the default env.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("sklearn")

from dphoney import distinguisher as dist  # noqa: E402
from dphoney.corpus import FORMATS, build_reference_corpus  # noqa: E402
from dphoney.generator import fit_dp_model, sample_canary, uniform_canary  # noqa: E402

SPEC = FORMATS["github_pat"]
N = 180


@pytest.fixture(scope="module")
def sources():
    reference = build_reference_corpus(seed=0, n_per_format=N)
    ref = reference.by_format["github_pat"]
    models = fit_dp_model(reference, epsilon=50.0, seed=0)
    rng = np.random.default_rng(123)
    dp = [sample_canary(models["github_pat"], rng) for _ in range(N)]
    template = [uniform_canary(SPEC, rng) for _ in range(N)]
    return ref, template, dp


def test_separability_scores_bounded(sources):
    ref, template, _ = sources
    sep = dist.evaluate_source(SPEC, ref, template)
    assert 0.0 <= sep.discriminator_auroc <= 1.0
    for value in sep.per_feature_auroc.values():
        assert 0.5 <= value <= 1.0


def test_headline_contrast_exists(sources):  # AE1
    ref, template, dp = sources
    res = dist.run_contrast(SPEC, ref, template, dp)
    # Template canaries are filterable; DP canaries are materially less so. At the
    # demo ε the DP source sits near chance, so the gap is large and robust.
    assert res.template.discriminator_auroc > 0.7
    assert res.template.discriminator_auroc - res.dp.discriminator_auroc > 0.2


def test_battery_positive_control_separates(sources):
    ref, _, _ = sources
    ctrl = dist.battery_positive_control(SPEC, ref)
    assert ctrl.passed and ctrl.auroc >= 0.9


def test_near_identical_distributions_near_chance():
    full = build_reference_corpus(seed=1, n_per_format=2 * N).by_format["github_pat"]
    a, b = full[:N], full[N:]
    sep = dist.evaluate_source(SPEC, a, b)
    assert sep.discriminator_auroc < 0.65


def test_left_half_control_distinguishes_structured_from_structureless():
    # Structured reference → template separates → control passes.
    structured = build_reference_corpus(seed=3, n_per_format=N).by_format["github_pat"]
    assert dist.left_half_control(SPEC, structured, seed=1).passed
    # Uniform "reference" has no structure → template (also uniform) can't be
    # told apart → the control must FAIL loudly (guards the false-null confound).
    rng = np.random.default_rng(7)
    structureless = [uniform_canary(SPEC, rng) for _ in range(N)]
    assert not dist.left_half_control(SPEC, structureless, seed=1).passed


def test_run_contrast_does_not_raise_when_close(sources):
    ref, _, dp = sources
    res = dist.run_contrast(SPEC, ref, dp, dp)
    assert 0.0 <= res.dp.discriminator_auroc <= 1.0
