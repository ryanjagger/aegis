"""DP-HONEY distinguisher battery (U3): how separable are canaries from real?

A battery of per-string features (character entropy, bigram log-likelihood under
a reference model, format validity, numeric-substring fraction) plus a
scikit-learn MLP discriminator trained to separate a canary source from the
synthetic reference class. Reported as held-out discriminator AUROC (≈0.5 =
indistinguishable; high = filterable) plus per-feature univariate AUROC.

The headline contrast: *template* (uniform) canaries should separate from the
structured reference, *DP* canaries should not. Two controls gate that reading:

  - **battery positive control** — a deliberately distinguishable (low-entropy)
    set the battery must separate, proving the discriminator pipeline works;
  - **headline left-half control** — template canaries must separate from the
    reference, proving the reference carries the structure the contrast needs.
    Without it, a structureless reference yields chance-vs-chance and reads as a
    false null.

Needs the opt-in ``dphoney`` group (scikit-learn). Also exposes
``bigram_loglik`` so the conformal layer (U4/U5) can reuse it as the
nonconformity score.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from dphoney.corpus import FormatSpec
from dphoney.generator import extract_bodies, fit_bigram_counts, uniform_canary

FEATURE_NAMES: tuple[str, ...] = ("entropy", "bigram_ll", "format_valid", "numeric_frac")


def reference_bigram_logp(spec: FormatSpec, strings: list[str], ridge: float = 1.0) -> np.ndarray:
    """Smoothed log transition probabilities of the reference class ``[V, V]``."""
    counts = fit_bigram_counts(spec, strings) + ridge
    probs = counts / counts.sum(axis=1, keepdims=True)
    return np.log(probs)


def _body(spec: FormatSpec, value: str) -> str:
    return "".join(extract_bodies(spec, value))


def _entropy(body: str, alphabet: str) -> float:
    if not body:
        return 0.0
    counts = np.array([body.count(c) for c in alphabet], dtype=float)
    p = counts[counts > 0] / counts.sum()
    return float(-(p * np.log2(p)).sum())


def bigram_loglik(spec: FormatSpec, value: str, ref_logp: np.ndarray) -> float:
    """Mean log-prob of ``value``'s body transitions under the reference model.

    Low for uniform/template strings (random transitions are improbable under a
    peaky reference), high for reference-like strings — the core contrast signal,
    and the nonconformity score the conformal layer wraps.
    """
    index = {c: i for i, c in enumerate(spec.alphabet)}
    total, n = 0.0, 0
    for body in extract_bodies(spec, value):
        for a, b in zip(body, body[1:], strict=False):
            total += ref_logp[index[a], index[b]]
            n += 1
    return total / n if n else 0.0


def _numeric_frac(body: str) -> float:
    return sum(c.isdigit() for c in body) / len(body) if body else 0.0


def featurize(spec: FormatSpec, values: list[str], ref_logp: np.ndarray) -> np.ndarray:
    feats = []
    for value in values:
        body = _body(spec, value)
        feats.append(
            [
                _entropy(body, spec.alphabet),
                bigram_loglik(spec, value, ref_logp),
                1.0 if spec.validate(value) else 0.0,
                _numeric_frac(body),
            ]
        )
    return np.array(feats, dtype=float)


@dataclass(frozen=True)
class Separability:
    discriminator_auroc: float
    discriminator_acc: float
    per_feature_auroc: dict[str, float]


def _univariate_auroc(labels: np.ndarray, values: np.ndarray) -> float:
    if len(set(labels)) < 2 or np.allclose(values, values[0]):
        return 0.5
    auc = roc_auc_score(labels, values)
    return float(max(auc, 1.0 - auc))


def evaluate_source(
    spec: FormatSpec,
    reference_values: list[str],
    candidate_values: list[str],
    *,
    ref_logp: np.ndarray | None = None,
    seed: int = 0,
) -> Separability:
    """Train an MLP to separate ``candidate_values`` from ``reference_values``.

    When ``ref_logp`` is not supplied, the reference bigram model is fit on a
    held-out half of ``reference_values`` and the *other* half is used as the
    class-0 examples, so the bigram-LL feature is out-of-sample for both classes.
    Scoring the class-0 set under a model fit on those same strings would inflate
    its likelihood and fabricate separation (an in-sample-bias confound).
    """
    ref_eval = reference_values
    if ref_logp is None:
        mid = len(reference_values) // 2
        ref_logp = reference_bigram_logp(spec, reference_values[:mid])
        ref_eval = reference_values[mid:]
    x_ref = featurize(spec, ref_eval, ref_logp)
    x_cand = featurize(spec, candidate_values, ref_logp)
    x = np.vstack([x_ref, x_cand])
    y = np.concatenate([np.zeros(len(x_ref)), np.ones(len(x_cand))])

    x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.4, random_state=seed, stratify=y)
    scaler = StandardScaler().fit(x_tr)
    clf = MLPClassifier(hidden_layer_sizes=(16,), max_iter=1000, random_state=seed)
    clf.fit(scaler.transform(x_tr), y_tr)
    proba = clf.predict_proba(scaler.transform(x_te))[:, 1]
    auroc = float(roc_auc_score(y_te, proba)) if len(set(y_te)) > 1 else 0.5
    acc = float(clf.score(scaler.transform(x_te), y_te))
    per_feature = {name: _univariate_auroc(y, x[:, i]) for i, name in enumerate(FEATURE_NAMES)}
    return Separability(auroc, acc, per_feature)


@dataclass(frozen=True)
class ContrastResult:
    template: Separability
    dp: Separability


def run_contrast(
    spec: FormatSpec,
    reference_values: list[str],
    template_values: list[str],
    dp_values: list[str],
    *,
    seed: int = 0,
) -> ContrastResult:
    """The headline: template-vs-reference and DP-vs-reference separability.

    The reference bigram model is fit on a held-out half so both the reference
    eval set and the candidates are scored out-of-sample (no in-sample bias).
    """
    mid = len(reference_values) // 2
    ref_logp = reference_bigram_logp(spec, reference_values[:mid])
    ref_eval = reference_values[mid:]
    return ContrastResult(
        template=evaluate_source(spec, ref_eval, template_values, ref_logp=ref_logp, seed=seed),
        dp=evaluate_source(spec, ref_eval, dp_values, ref_logp=ref_logp, seed=seed),
    )


@dataclass(frozen=True)
class ControlResult:
    name: str
    auroc: float
    passed: bool


def _low_entropy_value(spec: FormatSpec, rng: np.random.Generator) -> str:
    """Format-valid string with a 2-symbol body — clearly low-entropy."""
    sub = spec.alphabet[:2]
    parts = []
    for part in spec.parts:
        if isinstance(part, str):
            parts.append(part)
        else:
            parts.append("".join(sub[int(rng.integers(2))] for _ in range(part)))
    return "".join(parts)


def battery_positive_control(
    spec: FormatSpec,
    reference_values: list[str],
    *,
    seed: int = 0,
    floor: float = 0.9,
) -> ControlResult:
    """A deliberately distinguishable set the battery MUST separate.

    Low-entropy bodies are separable via the entropy feature regardless of
    whether the reference is structured, so this validates the discriminator
    pipeline independently of the left-half control.
    """
    rng = np.random.default_rng(seed)
    rigged = [_low_entropy_value(spec, rng) for _ in reference_values]
    sep = evaluate_source(spec, reference_values, rigged, seed=seed)
    return ControlResult(
        "battery_positive", sep.discriminator_auroc, sep.discriminator_auroc >= floor
    )


def left_half_control(
    spec: FormatSpec,
    reference_values: list[str],
    template_values: list[str] | None = None,
    *,
    seed: int = 0,
    floor: float = 0.6,
) -> ControlResult:
    """Template canaries must separate from the reference (the headline's left half).

    If they don't, the reference corpus lacks the bigram structure the contrast
    needs — fail loudly rather than reading the DP result as an honest null.
    """
    if template_values is None:
        rng = np.random.default_rng(seed)
        template_values = [uniform_canary(spec, rng) for _ in reference_values]
    sep = evaluate_source(spec, reference_values, template_values, seed=seed)
    return ControlResult(
        "left_half", sep.discriminator_auroc, sep.discriminator_auroc >= floor
    )
