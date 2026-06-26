"""Synthetic, format-valid credential corpus for the DP-HONEY lab (U1).

Pure data + numpy. No torch, no sklearn — importable from the base install so
the live injection path (U6) can reuse the format specs and generation without
the opt-in ``dphoney`` group.

The reference corpus is the lab's stand-in for *real* credentials: format-valid
strings whose body characters carry **non-uniform bigram structure** (sampled
from a seeded per-format Markov chain). That structure is the whole point of the
DP-HONEY contrast — a differentially-private bigram model fit on these strings
can reproduce it, while the repo's template generator
(``app/canaries/generator.py``, uniform ``secrets.choice``) cannot. Without
non-uniform structure here, template canaries would be in-distribution for the
reference class and the distinguisher contrast (U3) would collapse to
chance-vs-chance.
"""

from __future__ import annotations

import string
from dataclasses import dataclass

import numpy as np

ALNUM = string.ascii_letters + string.digits
UPPER_DIGITS = string.ascii_uppercase + string.digits
B64URL = string.ascii_letters + string.digits + "-_"

# A "part" is either a fixed literal (str) or a variable body segment length (int).
Part = str | int


@dataclass(frozen=True)
class FormatSpec:
    """One credential format: fixed literals interleaved with variable body runs.

    ``parts`` alternates literals (``str``) and body lengths (``int``). The body
    runs are the format's free-entropy region — what the bigram model shapes and
    what the DP generator (U2) samples; the literals are the fixed scaffolding it
    templates verbatim.
    """

    name: str
    parts: tuple[Part, ...]
    alphabet: str

    @property
    def body_segments(self) -> tuple[int, ...]:
        return tuple(p for p in self.parts if isinstance(p, int))

    @property
    def body_length(self) -> int:
        return sum(self.body_segments)

    @property
    def n_bigram_transitions(self) -> int:
        """ℓ1 sensitivity of the per-format bigram count vector.

        The number of within-segment character transitions one credential
        contributes when added to / removed from the fitting corpus. For a
        single-body format this is ``body_length - 1`` (the plan's
        ``max_length - 1``); for multi-segment formats it is the sum over
        segments, since transitions are not counted across the literal gaps.
        """
        return sum(max(seg - 1, 0) for seg in self.body_segments)

    def validate(self, value: str) -> bool:
        pos = 0
        alphabet = set(self.alphabet)
        for part in self.parts:
            if isinstance(part, str):
                if not value.startswith(part, pos):
                    return False
                pos += len(part)
            else:
                segment = value[pos : pos + part]
                if len(segment) != part or any(c not in alphabet for c in segment):
                    return False
                pos += part
        return pos == len(value)


# The six formats mirror app/canaries/generator.py so DP canaries are drop-in.
FORMATS: dict[str, FormatSpec] = {
    "github_pat": FormatSpec("github_pat", ("ghp_", 36), ALNUM),
    "stripe_key": FormatSpec("stripe_key", ("sk_live_", 32), ALNUM),
    "aws_access_key": FormatSpec("aws_access_key", ("AKIA", 16), UPPER_DIGITS),
    "postgres_url": FormatSpec(
        "postgres_url",
        ("postgres://ais_user:", 20, "@db.local:5432/ais_demo_", 6),
        ALNUM,
    ),
    "jwt_like": FormatSpec("jwt_like", (18, ".", 24, ".", 22), B64URL),
    "support_token": FormatSpec("support_token", ("support_live_", 28), ALNUM),
}

FORMAT_NAMES: tuple[str, ...] = tuple(FORMATS)


def bigram_profile(alphabet: str, seed: int) -> np.ndarray:
    """A deterministic, **non-uniform** first-order transition matrix ``[V, V]``.

    Dirichlet(0.3) rows are peaky, so the chain favours particular character
    transitions — the bigram structure the DP model reproduces and the
    distinguisher measures. Uniform template generation has no such structure.
    """
    rng = np.random.default_rng(seed)
    v = len(alphabet)
    return rng.dirichlet(np.full(v, 0.3), size=v)


def sample_body(
    alphabet: str, length: int, profile: np.ndarray, rng: np.random.Generator
) -> str:
    """Sample ``length`` characters by walking the ``profile`` Markov chain."""
    if length == 0:
        return ""
    v = len(alphabet)
    idx = int(rng.integers(v))
    out: list[str] = []
    for _ in range(length):
        out.append(alphabet[idx])
        idx = int(rng.choice(v, p=profile[idx]))
    return "".join(out)


def sample_structured(
    spec: FormatSpec, profile: np.ndarray, rng: np.random.Generator
) -> str:
    """One format-valid value: literals verbatim, body runs from the chain."""
    parts: list[str] = []
    for part in spec.parts:
        if isinstance(part, str):
            parts.append(part)
        else:
            parts.append(sample_body(spec.alphabet, part, profile, rng))
    return "".join(parts)


@dataclass(frozen=True)
class ReferenceCorpus:
    """Synthetic 'real credential' reference class, per format, plus its profiles."""

    by_format: dict[str, list[str]]
    profiles: dict[str, np.ndarray]

    def all_values(self) -> list[str]:
        return [v for values in self.by_format.values() for v in values]


def build_reference_corpus(seed: int = 0, n_per_format: int = 200) -> ReferenceCorpus:
    """Synthetic, format-valid reference strings with non-uniform bigram structure.

    Deterministic from ``seed``. Each format gets its own seeded Markov profile,
    so the reference distribution is structured (not uniform) and the DP-vs-
    template contrast in U3 is well-posed. No real credentials are read or
    generated; every value is synthesized here.
    """
    by_format: dict[str, list[str]] = {}
    profiles: dict[str, np.ndarray] = {}
    for fi, (name, spec) in enumerate(FORMATS.items()):
        profile = bigram_profile(spec.alphabet, seed=seed * 1000 + fi)
        rng = np.random.default_rng(seed * 1000 + fi + 1)
        profiles[name] = profile
        by_format[name] = [sample_structured(spec, profile, rng) for _ in range(n_per_format)]
    return ReferenceCorpus(by_format=by_format, profiles=profiles)
