"""End-to-end DP-HONEY lab (U5): ``uv run --group dphoney python -m dphoney.run``.

Orchestrates corpus -> DP fit -> template/DP generation -> distinguisher contrast
-> conformal calibration -> figures -> gated interpretation. Model-free (no LLM),
so the whole pipeline runs in well under a second on the synthetic corpus.

``build_interpretation`` is a pure function gated on the distinguisher controls,
so its honesty logic — refusing to read the contrast when a control failed — is
unit-tested without running the lab.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dphoney import figures
from dphoney.artifacts import (
    BIGRAM_MODEL_NPZ,
    CALIBRATION_JSON,
    COVERAGE_PNG,
    INTERPRETATION_MD,
    SEPARABILITY_PNG,
    artifacts_dir,
)
from dphoney.calibration import Calibration, calibrate, save_calibration
from dphoney.corpus import FORMATS, build_reference_corpus
from dphoney.distinguisher import (
    ControlResult,
    battery_positive_control,
    bigram_loglik,
    left_half_control,
    reference_bigram_logp,
    run_contrast,
)
from dphoney.generator import (
    DEFAULT_EPSILON,
    fit_dp_model,
    sample_canary,
    save_dp_model,
    uniform_canary,
)

HEADLINE_FORMAT = "github_pat"


def build_interpretation(
    battery: ControlResult,
    left_half: ControlResult,
    *,
    template_auroc: float,
    dp_auroc: float,
    epsilon: float,
    coverage: float,
    target_coverage: float,
) -> str:
    """Compose the written interpretation, gated on the distinguisher controls.

    A failed battery control means the discriminator pipeline is unvalidated, so
    the contrast is not interpretable. A failed left-half control means the
    reference lacks structure, so a low DP separability is a confound, not a real
    null. Only with both controls passing is the contrast read as a finding.
    """
    lines = ["# DP-HONEY lab interpretation", ""]
    lines.append(f"Battery control: **{'pass' if battery.passed else 'FAIL'}** "
                 f"(AUROC {battery.auroc:.3f}).")
    lines.append(f"Left-half control: **{'pass' if left_half.passed else 'FAIL'}** "
                 f"(AUROC {left_half.auroc:.3f}).")
    lines.append("")

    if not battery.passed:
        lines += [
            "The battery positive control did **not** separate a must-separate set, so the "
            "distinguisher pipeline is not validated. The contrast below is **not "
            "interpretable** — fix the harness (features, scaler, MLP) before drawing any "
            "conclusion.",
            "",
        ]
        return "\n".join(lines)

    if not left_half.passed:
        lines += [
            "The headline left-half control failed: template (uniform) canaries did **not** "
            "separate from the reference, so the reference corpus lacks the bigram structure "
            "the contrast needs. A low DP separability here is **not a real null** — it is a "
            "structureless-reference confound. Rebuild the reference with non-uniform "
            "structure.",
            "",
        ]
        return "\n".join(lines)

    gap = template_auroc - dp_auroc
    lines.append(
        f"Template canaries separate from real-format credentials (mean AUROC "
        f"{template_auroc:.3f}); DP canaries are materially less separable (mean AUROC "
        f"{dp_auroc:.3f}, gap {gap:+.3f}). The DP bigram model reproduces the reference "
        "structure the uniform template generator lacks, so a distinguisher that filters "
        "template canaries fails on DP canaries."
    )
    lines.append("")
    lines.append(
        f"Generated at ε = {epsilon:g} (per-format, on the released bigram-count model). "
        "This is a deliberately weak DP guarantee, tuned for a legible contrast on a small "
        "synthetic corpus — **not** a claim of indistinguishability from real credentials. "
        "A larger corpus shifts the frontier toward smaller ε."
    )
    lines.append("")
    lines.append(
        f"Conformal calibration: coverage {coverage:.3f} on held-out benign (target "
        f"{target_coverage:.3f}) with no hand-tuning. Conformal sets the benign "
        "false-positive rate α only; the miss rate β in Pr(detect)=k/(m+k)(1−β) is an "
        "independent detection property the threshold does not set."
    )
    lines.append("")
    return "\n".join(lines)


@dataclass(frozen=True)
class LabResult:
    per_format: dict[str, tuple[float, float]]  # name -> (template_auroc, dp_auroc)
    battery: ControlResult
    left_half: ControlResult
    calibration: Calibration
    epsilon: float

    @property
    def template_auroc(self) -> float:
        return float(np.mean([t for t, _ in self.per_format.values()]))

    @property
    def dp_auroc(self) -> float:
        return float(np.mean([d for _, d in self.per_format.values()]))


def run_lab(
    seed: int = 0,
    n_per_format: int = 200,
    epsilon: float = DEFAULT_EPSILON,
    alpha: float = 0.01,
    n_calibration: int = 300,
) -> LabResult:
    """Run the full contrast + controls + conformal calibration."""
    reference = build_reference_corpus(seed=seed, n_per_format=n_per_format)
    models = fit_dp_model(reference, epsilon=epsilon, seed=0)
    rng = np.random.default_rng(123)

    per_format: dict[str, tuple[float, float]] = {}
    for name, spec in FORMATS.items():
        ref = reference.by_format[name]
        template = [uniform_canary(spec, rng) for _ in ref]
        dp = [sample_canary(models[name], rng) for _ in ref]
        contrast = run_contrast(spec, ref, template, dp, seed=seed)
        per_format[name] = (
            contrast.template.discriminator_auroc,
            contrast.dp.discriminator_auroc,
        )

    head_spec = FORMATS[HEADLINE_FORMAT]
    head_ref = reference.by_format[HEADLINE_FORMAT]
    battery = battery_positive_control(head_spec, head_ref, seed=seed)
    left_half = left_half_control(head_spec, head_ref, seed=seed)

    # Conformal calibration over the REAL nonconformity score (negated bigram-LL,
    # so real-credential-like strings are conforming) on held-out benign folds.
    # A dedicated, larger benign draw lets conformal resolve the 1-α tail (you
    # need on the order of 1/α calibration points).
    big = build_reference_corpus(seed=seed, n_per_format=3 * n_calibration)
    benign = big.by_format[HEADLINE_FORMAT]
    fit_fold = benign[:n_calibration]
    calib_fold = benign[n_calibration : 2 * n_calibration]
    eval_fold = benign[2 * n_calibration :]
    ref_logp = reference_bigram_logp(head_spec, fit_fold)
    calib_scores = [-bigram_loglik(head_spec, v, ref_logp) for v in calib_fold]
    eval_scores = [-bigram_loglik(head_spec, v, ref_logp) for v in eval_fold]
    calibration = calibrate(calib_scores, eval_scores, alpha=alpha)

    return LabResult(per_format, battery, left_half, calibration, epsilon)


def main() -> None:  # pragma: no cover - CLI entrypoint
    out = artifacts_dir()
    result = run_lab()

    save_dp_model(
        fit_dp_model(
            build_reference_corpus(seed=0, n_per_format=200), epsilon=result.epsilon, seed=0
        ),
        out / BIGRAM_MODEL_NPZ,
    )
    save_calibration(result.calibration, out / CALIBRATION_JSON)
    figures.plot_separability_contrast(result.per_format, out / SEPARABILITY_PNG)
    target = 1.0 - result.calibration.alpha
    figures.plot_coverage(
        result.calibration.naive_coverage, result.calibration.coverage, target, out / COVERAGE_PNG
    )
    interpretation = build_interpretation(
        result.battery,
        result.left_half,
        template_auroc=result.template_auroc,
        dp_auroc=result.dp_auroc,
        epsilon=result.epsilon,
        coverage=result.calibration.coverage,
        target_coverage=target,
    )
    (out / INTERPRETATION_MD).write_text(interpretation)
    print(interpretation)
    print(f"\nArtifacts written to {out}")


if __name__ == "__main__":  # pragma: no cover
    main()
