---
date: 2026-06-26
topic: dphoney-pillar-and-demo-integration
---

# DP-HONEY Pillar + AIS Demo Integration — Requirements

## Summary

Build the paper's second research pillar, DP-HONEY (calibrated honeytokens),
for real: a differentially-private character model that generates format-matched
canaries, a distinguisher test showing they resist filtering better than today's
template canaries, and split-conformal calibration that sets the detector
threshold without hand-tuning. Then stitch the existing surfaces into one guided
demo that walks the full AIS pipeline (DP-HONEY → CIFT → text backstop →
NIMBUS), feeding DP canaries into the live injection path and presenting CIFT as
an opt-in white-box pre-output station.

---

## Problem Frame

The repo implements two of the paper's three pillars: the I/O half (canary
injection, transform-aware scanning, tool-call interception, heuristic
NIMBUS-lite) and CIFT (activation probing, lab plus live monitor). Two things are
missing for a demo that "covers the paper": DP-HONEY, the calibrated-honeytoken
pillar; and a single narrative that shows the four monitors working as one system
(the paper's Figure 1) instead of three disconnected surfaces.

The canary surface is both the weakest piece and the cheapest real pillar to add.
Today's canaries are template-generated — exactly the kind the paper argues an
attacker can filter by prefix, checksum, entropy, or length. DP-HONEY replaces
that with canaries drawn from a fitted statistical model, and it is pure
statistics: no model in the loop, no torch, no MPS. It also pairs with CIFT as a
contrast in defensive philosophy — black-box deception versus white-box
monitoring. The remaining gap is integration: CIFT lives in its own Streamlit
app, the live pipeline runs on a mock adapter, and nothing tells the end-to-end
story in one sitting.

---

## Key Decisions

- **DP-HONEY is the second pillar; real InfoNCE NIMBUS is not built.** DP-HONEY
  is lighter (numpy/sklearn, no torch/MPS), adds a genuinely new and showable
  surface, and pairs with CIFT as a contrast in defensive philosophy. Real
  InfoNCE NIMBUS would be a heavier, model-in-the-loop build whose headline
  visual the heuristic NIMBUS-lite already produces, so it is deferred.

- **DP-HONEY's headline is a contrast result, mirroring CIFT's.** Just as CIFT
  shows activation detection surviving an encoding that defeats text matching,
  DP-HONEY shows a distinguisher separating *template* canaries from real-format
  credentials while *failing* to separate *DP* canaries. The contrast is the
  point — "the filterable baseline fails here, ours holds" is stronger than a
  lone "our canaries look real" number.

- **A weak or null DP-HONEY result is a documented finding, not a failure.** On a
  small synthetic corpus the DP canaries may stay partly distinguishable.
  Following the repo's established honesty stance (CIFT's), the requirement is a
  correct, interpreted measurement — not a target separability number.

- **The distinguisher's "real" class is synthetic, format-valid strings.** No
  real credentials enter the repo. The reference distribution the distinguisher
  tries to separate canaries from is a held-out set of format-valid synthetic
  credentials.

- **Integration is a stitched narrative, not a fully-live pipeline.** One guided
  walkthrough reuses existing surfaces: a DP-HONEY view, the fast mock-backed
  live pipeline (DP canary inject → text scan → NIMBUS), and the CIFT live
  monitor as the pre-output station. CIFT's model load stays isolated and opt-in
  (the existing `cift` dependency-group pattern); the default pipeline stays
  fast. The cost is that CIFT does not gate literally inline in `/v1/responses`
  per the paper's Algorithm 1.

- **DP canaries plug into the live injection path behind a toggle.** The
  injection path can emit either template or DP canaries, so the demo can show
  the same distinguisher that filters a template canary fail on the DP one.
  NIMBUS-lite remains and is labeled honestly as a heuristic stand-in for the
  paper's InfoNCE estimator.

---

## Requirements

### DP-HONEY: generation

- R1. A per-format character-bigram model is fit for each credential format the
  repo already uses (github_pat, stripe_key, aws_access_key, postgres_url,
  jwt_like, support_token), from a corpus of synthetic format-valid strings.
- R2. Laplace noise is added to the bigram count table to provide an ε-DP
  guarantee for the released model, with ε configurable. Generation draws
  canaries from the noised model and enforces format validity (length, charset,
  required prefix/structure).
- R3. The DP generator is exposed as a canary source the existing injection path
  can use in place of template generation, selectable at inject time so
  template-vs-DP can be compared in one session.

### DP-HONEY: distinguisher evaluation (the contrast)

- R4. A distinguisher battery scores how separable canaries are from the
  synthetic real-format reference class: character-entropy, bigram
  log-likelihood, format validation, numeric-substring tests, and a small
  discriminator MLP. It reports a separability metric (e.g., discriminator
  AUROC/accuracy) per canary source.
- R5. The same battery runs on template canaries and DP canaries against the
  reference class, producing the headline contrast: template canaries separate
  materially above chance while DP canaries sit at or near chance. A result where
  DP canaries remain separable is recorded and interpreted, not treated as a
  failed requirement.
- R6. The reference "real credential" class is synthetic and format-valid; no
  real credentials are used anywhere.

### DP-HONEY: conformal calibration

- R7. The canary detector threshold is set by split-conformal calibration: given
  a held-out benign calibration set and target significance α, the threshold is
  taken from the empirical distribution of benign nonconformity scores, removing
  manual threshold tuning.
- R8. Calibration reports empirical coverage against the target (e.g., 0.99) on a
  separate held-out benign set, alongside the no-conformal baseline for contrast.

### DP-HONEY: canary accounting

- R9. The canary-accounting relation Pr(detect) = k/(m+k)·(1−β) is surfaced
  (e.g., a small interactive calculator or visual) so the demo can show how
  detection probability depends on planted-canary count k, real-credential count
  m, and detector miss rate β.

### Demo integration

- R10. A single guided demo entry point walks the full AIS pipeline in order —
  DP-HONEY (calibrated canaries) → CIFT (pre-output activation gate) → text-level
  canary backstop → NIMBUS (cumulative ledger) — reusing the existing surfaces
  rather than rebuilding them.
- R11. The live `/v1/responses` pipeline injects DP canaries (per R3) so the
  integrated demo exercises calibrated honeytokens end to end; the default
  backend path stays mock-adapter and fast.
- R12. CIFT appears as an opt-in white-box "pre-output" station whose model load
  is isolated to that station (the existing `cift` group pattern); it is not
  required for the default pipeline to run.
- R13. NIMBUS is presented and labeled as the heuristic NIMBUS-lite, explicitly
  distinguished from the paper's InfoNCE estimator, so the demo's honesty about
  what is and isn't "real research" is preserved.

### Learning artifact

- R14. A short written interpretation accompanies the DP-HONEY figures: the
  separability observed for template vs DP canaries, the ε used, the conformal
  coverage achieved, and the likely cause of any residual separability (corpus
  size, format constraints, ε).

---

## Key Flows

- F1. DP-HONEY lab (per credential format)
  - **Trigger:** The DP-HONEY lab is run over the supported credential formats.
  - **Steps:** Build synthetic format-valid corpus → fit per-format bigram model
    → add Laplace noise (ε) → generate DP canaries → run distinguisher battery on
    template and DP canaries vs the reference class → conformal-calibrate the
    detector → emit figures + interpretation.
  - **Outcome:** A separability contrast (template vs DP), a conformal coverage
    number, and a written interpretation.
  - **Covers:** R1, R2, R4, R5, R7, R8, R14.

- F2. Integrated demo walkthrough (per request)
  - **Trigger:** A presenter runs a scenario through the stitched walkthrough.
  - **Steps:** DP canary injected into context → (opt-in) CIFT scores readout
    activations pre-output → output generated (mock adapter by default) → text
    scanner checks for canary leakage → NIMBUS-lite updates the cumulative
    ledger.
  - **Outcome:** One screen-to-screen narrative covering all four monitors.
  - **Covers:** R10, R11, R12, R13.

---

## Acceptance Examples

- AE1. Calibrated-canary contrast
  - **Covers R4, R5.**
  - **Given** a discriminator trained to separate canaries from synthetic
    real-format credentials,
  - **When** it scores template canaries and DP canaries,
  - **Then** it separates the template canaries materially above chance while the
    DP canaries score at or near chance. If the DP canaries are also separable on
    this corpus, that is recorded as a finding, not a failed requirement.

- AE2. Conformal calibration removes hand-tuning
  - **Covers R7, R8.**
  - **Given** a held-out benign calibration set and target significance α,
  - **When** the detector threshold is set by split-conformal calibration,
  - **Then** empirical coverage on a separate benign set lands near the target
    with no manual threshold tuning.

- AE3. DP canaries flow through the live pipeline
  - **Covers R3, R11.**
  - **Given** the live pipeline configured to inject DP canaries,
  - **When** a scenario runs,
  - **Then** the injected canary is a DP-generated, format-valid value and the
    text backstop still detects it on a leak.

---

## Success Criteria

- The DP-HONEY lab runs end to end and produces the template-vs-DP separability
  contrast and a conformal coverage number.
- You can explain the result in your own words: how the DP model lowers
  separability, what ε bought, and why conformal calibration beats a hand-set
  threshold.
- A weak or null DP-HONEY result counts as success when it is interpreted and the
  template baseline *is* separable — so the explanation rests on DP-HONEY's
  behavior, not a harness bug.
- The integrated walkthrough tells the four-monitor story in one sitting, with DP
  canaries live in the pipeline and NIMBUS honestly labeled.

---

## Scope Boundaries

### Deferred for later

- Real InfoNCE NIMBUS (learned bits-based cumulative estimator) — its own pillar,
  deferred in favor of DP-HONEY.
- Evasion-suite breadth (Unicode homoglyphs, leet, fragmentation, paraphrase,
  reverse, partial reproduction) beyond what CIFT and the text scanner cover.
- Multi-session leakage state (the paper's "restart session to reset the budget"
  attack) — NIMBUS stays per-session.
- Public prompt-injection benchmarks (TensorTrust, InjecAgent, BIPIA, AgentDojo)
  — the corpus stays authored/synthetic.
- A fully-live single pipeline where every request runs the white-box model so
  CIFT gates inline per Algorithm 1.

### Outside this effort

- Real (non-fake) credentials, cloud/GPU training, deployment, multi-user
  hosting, product/UI polish. The repo's standing constraint holds: all
  credentials fake, all tools local.
- The stubbed live model adapters (OpenAI, Ollama) stay stubbed.

---

## Dependencies / Assumptions

- **No new heavy deps.** DP-HONEY needs only numpy (base) plus scikit-learn and
  matplotlib, already in the optional `cift` group; no torch/MPS. Generation and
  calibration are pure statistics.
- **Synthetic corpora are authored.** The per-format fitting corpus and the
  distinguisher's reference class are synthetic, format-valid strings authored
  for the lab; the benign calibration set for conformal can reuse or extend the
  existing benign prompt corpus.
- **Behavioral assumption (load-bearing).** A per-format character-bigram model
  fit on a modest synthetic corpus produces canaries realistic enough that the
  distinguisher's separability drops meaningfully versus template canaries. If it
  does not, the first response is to enrich the fitting corpus or relax
  over-rigid format constraints — not to abandon the contrast — and a residual
  gap is a documented finding.
- **DP semantics are scoped honestly.** ε-DP applies to the released
  bigram-count model with respect to neighboring credential corpora; it is not a
  claim that canaries are computationally indistinguishable from real
  credentials. The interpretation states this.

---

## Outstanding Questions

### Deferred to Planning

- Where DP-HONEY lives: a new `dphoney/` lab package mirroring `cift/` plus a
  generator that plugs into `app/canaries/`, vs. extending `app/canaries/`
  directly.
- The unified walkthrough's form: a new top-level Streamlit "AIS walkthrough" app
  vs. added tabs/launcher over the existing dashboard plus CIFT monitor.
- Whether the integrated demo also adds a thin live CIFT gate hook in the
  pipeline (active only in white-box mode) or keeps CIFT purely as the separate
  monitor station.
- Which ε value(s) to demo, and whether to show an ε sweep (privacy/utility
  tradeoff) à la the paper's budget-sensitivity figure.
- How many synthetic strings per format to fit the bigram model and to form the
  distinguisher's reference and calibration sets.

---

## Sources / Research

- Paper `docs/2606.04141v1.pdf` — §4.3 DP-HONEY (character bigram + Laplace ε-DP,
  conformal calibration, distinguisher tests, canary-accounting
  Pr(detect)=k/(m+k)(1−β)), Figure 1 (four-monitor architecture), §4.1 +
  Algorithm 1 (integrated pipeline), §6 (tool-call and multi-session
  limitations).
- Prior brainstorm `docs/brainstorms/2026-06-25-cift-activation-probe-lab-requirements.md`
  — the CIFT pillar; established the honesty/contrast stance this doc mirrors.
- Existing canary surface to upgrade: `app/canaries/generator.py` (template
  generation), `app/canaries/injector.py` (injection path),
  `app/scanners/canary_scanner.py` (detector to calibrate).
- CIFT surfaces to stitch in: `cift/live_monitor.py` (pre-output station),
  `cift/run.py` (lab pipeline pattern to mirror for DP-HONEY).
- Heuristic NIMBUS to label: `app/nimbus/scoring.py`, `app/nimbus/ledger.py`.
</content>
</invoke>
