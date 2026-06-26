"""DP-HONEY generator (U2): per-format character-bigram model + Laplace ε-DP.

Fits a transition-count model on the *body* characters of reference credentials,
adds Laplace noise to the released counts for an ε-DP guarantee, and samples
format-valid canaries from the noised model. Pure numpy — importable from the
base install so the live path (U6) can generate DP canaries without the opt-in
group.

Sensitivity. Under the credential-level neighbouring relation (add/remove one
credential), one credential contributes up to L-1 within-body bigram
transitions, so the ℓ1 sensitivity of the per-format transition-count vector is
``spec.n_bigram_transitions`` (== max_length-1 for single-body formats), NOT 1.
The Laplace scale is ``sensitivity / ε``. ε-DP is on the released count model,
not a claim of indistinguishability from real credentials.

Sampling is total by construction: the fixed prefix/scaffolding is templated and
only the free-entropy body is sampled from the alphabet, so every canary is
format-valid. ``sample_canary`` still validates and falls back defensively.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from dphoney.corpus import FORMATS, FormatSpec, ReferenceCorpus, build_reference_corpus

# Demo default, tuned so the contrast is legible on the small synthetic corpus:
# the per-credential ℓ1 sensitivity (~15-61) over a few-hundred-string corpus
# forces a privacy/utility frontier. At ε≈50 the noised model still reproduces
# the reference bigram structure (DP canaries indistinguishable, AUROC ≈ 0.5)
# while smaller ε collapses it toward uniform (DP becomes as filterable as
# template). This is a weak DP guarantee by design — the demo shows the
# mechanism; a larger corpus shifts the frontier toward smaller ε. The lab
# reports the effective ε honestly.
DEFAULT_EPSILON = 50.0


def extract_bodies(spec: FormatSpec, value: str) -> list[str]:
    """Inverse of corpus.sample_structured: pull the body runs out of a value."""
    bodies: list[str] = []
    pos = 0
    for part in spec.parts:
        if isinstance(part, str):
            pos += len(part)
        else:
            bodies.append(value[pos : pos + part])
            pos += part
    return bodies


def fit_bigram_counts(spec: FormatSpec, strings: list[str]) -> np.ndarray:
    """Transition counts ``[V, V]`` over the body characters of ``strings``."""
    v = len(spec.alphabet)
    index = {c: i for i, c in enumerate(spec.alphabet)}
    counts = np.zeros((v, v), dtype=np.float64)
    for value in strings:
        for body in extract_bodies(spec, value):
            for a, b in zip(body, body[1:], strict=False):
                counts[index[a], index[b]] += 1.0
    return counts


def laplace_scale(sensitivity: float, epsilon: float) -> float:
    return sensitivity / epsilon


def add_laplace_noise(
    counts: np.ndarray, epsilon: float, sensitivity: float, rng: np.random.Generator
) -> np.ndarray:
    """Add ``Laplace(0, sensitivity/ε)`` to every count cell, clipped at zero."""
    noised = counts + rng.laplace(0.0, laplace_scale(sensitivity, epsilon), size=counts.shape)
    return np.clip(noised, 0.0, None)


def to_transition_probs(counts: np.ndarray) -> np.ndarray:
    """Row-normalize, smoothing all-zero rows to uniform (no divide-by-zero)."""
    v = counts.shape[1]
    rows = counts.sum(axis=1, keepdims=True)
    probs = np.where(rows > 0, counts / np.where(rows > 0, rows, 1.0), 1.0 / v)
    # Renormalize so each row sums to exactly 1.0 within float tolerance.
    return probs / probs.sum(axis=1, keepdims=True)


@dataclass(frozen=True)
class DPModel:
    spec: FormatSpec
    trans_probs: np.ndarray
    epsilon: float
    sensitivity: int


def _sample_body(spec: FormatSpec, trans_probs: np.ndarray, length: int, rng) -> str:
    if length == 0:
        return ""
    v = len(spec.alphabet)
    idx = int(rng.integers(v))
    out: list[str] = []
    for _ in range(length):
        out.append(spec.alphabet[idx])
        row = trans_probs[idx]
        idx = int(rng.choice(v, p=row / row.sum()))
    return "".join(out)


def uniform_canary(spec: FormatSpec, rng) -> str:
    """Uniform-random body, guaranteed format-valid.

    This is the distribution the repo's template generator
    (``app/canaries/generator.py``, ``secrets.choice``) draws from, so it doubles
    as the "template" baseline for the U3 contrast and as the defensive fallback
    when a modelled sample somehow fails validation.
    """
    parts: list[str] = []
    for part in spec.parts:
        if isinstance(part, str):
            parts.append(part)
        else:
            parts.append("".join(spec.alphabet[int(rng.integers(len(spec.alphabet)))]
                                 for _ in range(part)))
    return "".join(parts)


def sample_canary(model: DPModel, rng, max_retries: int = 4) -> str:
    """One format-valid DP canary: literals templated, body runs from the chain.

    Total by construction; the retry+fallback exists defensively so a degenerate
    model can never raise or loop unboundedly in the live path.
    """
    spec = model.spec
    for _ in range(max_retries + 1):
        parts: list[str] = []
        for part in spec.parts:
            if isinstance(part, str):
                parts.append(part)
            else:
                parts.append(_sample_body(spec, model.trans_probs, part, rng))
        value = "".join(parts)
        if spec.validate(value):
            return value
    return uniform_canary(spec, rng)


def fit_dp_model(
    reference: ReferenceCorpus | dict[str, list[str]],
    epsilon: float = DEFAULT_EPSILON,
    seed: int = 0,
) -> dict[str, DPModel]:
    """Fit a Laplace-noised bigram model per format. Seeded → reproducible.

    Seeding the noise draw makes the released model deterministic for a given
    (reference, ε, seed), so the live path and the lab share the exact model.

    Demo caveat: a fixed, publicly-known seed makes the ε-DP guarantee
    distributional only — an adversary who knows the seed can regenerate the
    noise and recover the raw counts. That is harmless here (the "protected"
    corpus is itself synthetic), but a real deployment over real credentials MUST
    draw fresh, secret noise per release or the guarantee is vacuous.
    """
    by_format = reference.by_format if isinstance(reference, ReferenceCorpus) else reference
    models: dict[str, DPModel] = {}
    for fi, (name, strings) in enumerate(by_format.items()):
        spec = FORMATS[name]
        counts = fit_bigram_counts(spec, strings)
        sensitivity = spec.n_bigram_transitions
        rng = np.random.default_rng(seed * 1000 + fi)
        noised = add_laplace_noise(counts, epsilon, sensitivity, rng)
        models[name] = DPModel(
            spec=spec,
            trans_probs=to_transition_probs(noised),
            epsilon=epsilon,
            sensitivity=sensitivity,
        )
    return models


def save_dp_model(models: dict[str, DPModel], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {f"{name}__trans": m.trans_probs for name, m in models.items()}
    meta = {
        name: {"epsilon": m.epsilon, "sensitivity": m.sensitivity}
        for name, m in models.items()
    }
    np.savez(path, **arrays)
    path.with_suffix(".json").write_text(json.dumps(meta))
    return path


def load_dp_model(path: str | Path) -> dict[str, DPModel]:
    path = Path(path)
    data = np.load(path)
    meta = json.loads(path.with_suffix(".json").read_text())
    return {
        name: DPModel(
            spec=FORMATS[name],
            trans_probs=data[f"{name}__trans"],
            epsilon=float(info["epsilon"]),
            sensitivity=int(info["sensitivity"]),
        )
        for name, info in meta.items()
    }


# --- Live-path helper: the exact default-ε model the lab measures ----------------

_DEFAULT_MODELS: dict[str, DPModel] | None = None


def default_dp_models() -> dict[str, DPModel]:
    """Seeded default-ε models on a fixed reference corpus, cached per process.

    The live injection path (U6) draws from these so its DP canaries come from
    the exact model the lab scores — same corpus, ε, and noise realization.
    """
    global _DEFAULT_MODELS
    if _DEFAULT_MODELS is None:
        reference = build_reference_corpus(seed=0, n_per_format=200)
        _DEFAULT_MODELS = fit_dp_model(reference, epsilon=DEFAULT_EPSILON, seed=0)
    return _DEFAULT_MODELS


def sample_default_canary(format_name: str, rng=None) -> str:
    rng = rng if rng is not None else np.random.default_rng()
    return sample_canary(default_dp_models()[format_name], rng)
