---
date: 2026-06-25
topic: cift-activation-probe-lab
---

# CIFT Activation-Probe Lab — Requirements

## Summary

Build an offline probe lab that reproduces the paper's CIFT result on a small
local model: hook the last transformer layers, pool readout-position
activations, fit an unsupervised Mahalanobis detector (with a learned probe as a
conditional add), and show that activation-level detection holds flat under an
encoding requested in-prompt where the existing text scanner collapses. Wiring
the trained detector into the live `/v1/responses` pipeline as a pre-output gate
is an explicit stretch goal, taken only if the lab result is strong.

---

## Problem Frame

The repo (AIS-IO) is a complete, working implementation of the *I/O half* of the
AIS paper — canary injection, transform-aware scanning, tool-call interception,
and a heuristic NIMBUS-lite. The three hardest, most research-interesting pieces
the paper actually claims are all still unbuilt: CIFT (activation probes),
DP-HONEY (calibrated honeytokens), and the real InfoNCE NIMBUS. That is the fork
the project is stuck at.

The goal of "where next" is **learning** — the win is understanding a mechanism,
and building is in service of that. CIFT is the highest-leverage target: it is
the paper's most surprising claim (catching credential *access* before any token
is emitted, AUROC 0.998), and it is the one piece that cannot be understood from
a notebook write-up or a blog post without building it. It also forces a real
local model into the loop, which makes the project stop running on mock data.

The cost shape today: the model layer is entirely placeholder
(`app/models/base.py` defines the protocol; `app/models/ollama_model.py` raises;
no white-box adapter exists), and there is no ML stack — only `numpy` is present
transitively. So this is genuinely new ground, not a refactor.

---

## Key Decisions

- **Lab-first; live wire-in is a stretch.** The offline experiment is the
  primary deliverable. For a learning goal, mechanism density beats integration
  plumbing, and the lab is a complete, satisfying outcome on its own.

- **The centerpiece is the text-vs-activation contrast — but it only holds if
  set up correctly.** The insight under test is that the model accesses the
  secret *before* it encodes the output, so internal features survive encodings
  that defeat text matching. Two setup conditions are non-negotiable: the
  encoding must be requested *in the model-visible prompt* (so the input — and
  thus CIFT's readout activations — actually differ across encodings; activations
  are input-side, encoding is an output transform), and the text-scanner baseline
  must use an encoding it does not already normalize. The existing
  `transforms.py` decodes base64 and hex, so those cannot show "text collapse";
  rot13 or a simple substitution cipher can.

- **Author the prompt corpus from scratch, designed to isolate credential
  access.** The repo's "scenarios" are mock output-template selectors, not
  steering prompts, so there is nothing to reuse. The experiment needs real
  injection prompts that induce credential *use*, benign prompts matched on
  length and vocabulary, and a negative-confound control — otherwise the probe
  may learn a surface feature (length, attack vocabulary) rather than the
  mechanism.

- **Small local model, default Qwen2.5-1.5B-Instruct.** Runs on Apple Silicon
  via MPS with no cloud. Accepts a possibly-weaker result than the paper's 7B in
  exchange for a tight local loop. Start smaller (0.5B) only if iteration speed
  hurts.

- **Pool at readout positions after the payload — never at static credential
  tokens.** This is the correctness crux. Because of causal masking, credential
  tokens cannot attend to the later injection, so probing them only asks "is a
  secret in context?" (trivial). Probing the readout position asks "is the model
  being steered to *use* it?" — the real claim. Getting this wrong yields a
  fake-good result.

- **Staged detector: Mahalanobis baseline is core; the learned probe is
  conditional.** The unsupervised per-layer Mahalanobis deviation is the core
  detector and ships end-to-end. The supervised probe — CFS-weighted (Causal Flow
  Score: the softplus-weighted per-layer scores) feeding a small MLP — is
  pursued only if the baseline plus contrast leave "do combined layers beat the
  best single layer?" open, or the baseline underperforms.

- **Validate before interpreting; then success is interpretation, not a
  number.** A positive control that *must* separate guards against reading a
  harness bug as a null result. On a 1.5B model the result may fall short of
  0.998 AUROC, or be null — and once the positive control passes and the
  surface-confound is ruled out, explaining the separation observed (or its
  absence) is the learning win.

---

## Requirements

### Model and activation extraction

- R1. A white-box adapter loads a small local instruct model (default
  Qwen2.5-1.5B-Instruct) and runs a forward pass over the injected-context +
  query prompt with per-layer hidden states accessible.
- R2. The adapter captures hidden states from the last K = ⌊0.25·L⌋ transformer
  layers (forward hooks or equivalent), where L is the model's layer count.
- R3. Features are pooled over readout positions occurring *after* both the
  secret context and the query/payload — at minimum the final prompt token and
  the first pre-generation decision position. Pooling at static credential-token
  positions is explicitly disallowed.
- R4. Extraction yields a per-layer feature vector (mean over readout positions)
  in a form the detector consumes.

### Prompt corpus and controls

- R5. A net-new labeled corpus is authored: real injection prompts that steer
  the model to *use* the injected secret, plus benign prompts. The repo's
  existing "scenarios" are mock output-template selectors, not steering prompts,
  and supply no reusable attack prompts.
- R6. The injected context used for the CIFT experiment omits the "these are
  fake canaries" framing and uses payloads that genuinely induce credential use.
  A behavioral check confirms the model actually attempts the leak before its
  activation scores are trusted.
- R7. Benign and credential-seeking prompts are matched on length and surface
  vocabulary, differing only in steering semantics. The set includes a
  negative-confound control: a benign prompt carrying the same exfiltration
  vocabulary but no real steer, which CIFT must *not* flag.
- R8. The held-out encoding suite uses an encoding the existing text scanner
  does not normalize (rot13 or a simple substitution cipher — base64 and hex are
  already decoded by `transforms.py` and cannot show text collapse), and the
  encoding is requested in the model-visible prompt so the input, and thus the
  readout activations, differ across encodings. The suite is held out entirely
  from detector fitting.

### Detector

- R9. A benign baseline (mean, covariance) is fit from benign prompts, and a
  per-layer Mahalanobis deviation is computed with diagonal covariance plus a
  ridge term. This unsupervised baseline is the core detector and runs end-to-end
  before any learned component.
- R10. A positive control validates the extraction-and-probe pipeline on a case
  that *must* separate — a deliberately blatant credential-use prompt, a public
  activation-probe dataset, or the paper's model/scenario — and must pass before
  any null result is interpreted.
- R11. The Mahalanobis detector reports AUROC, F1, and FPR on a held-out split.

### Evaluation and the contrast

- R12. The same held-out encoded scenarios run through both the existing
  text/transform scanner (against the undecoded encoding of R8) and CIFT,
  producing per-encoding detection scores for each so the two can be compared
  directly.
- R13. The lab produces two core figures: per-layer Mahalanobis deviation (benign
  vs attack) and the encoding-robustness comparison (text scanner vs CIFT).

### Learning artifact

- R14. A short written interpretation accompanies the figures: what separation
  was observed, which layers carried signal, and the likely cause (model size,
  readout-position choice, calibration-set size). A weak or null result is a
  valid, documented outcome — not a failure — provided the R10 positive control
  passed.

### Conditional: learned probe

- R15. (Conditional) A learned probe adds CFS-weighted (Causal Flow Score:
  softplus-weighted) per-layer scores combined into a small MLP detection logit,
  trained with binary cross-entropy plus a sparsity penalty, reporting its own
  AUROC/F1/FPR and the AUROC-by-individual-layer vs learned-ensemble figure.
  Pursued only if the Mahalanobis baseline plus contrast leave the
  "do combined layers beat the best single layer?" question open, or the baseline
  underperforms (see AE3's kill criterion).

### Stretch: live wire-in

- R16. (Stretch) The trained detector is exposed as a pre-output gate in
  `/v1/responses` that scores before generation and routes/aborts on a flag,
  surfaced in the dashboard alongside canary and NIMBUS traces. Taken only if
  the lab AUROC is strong enough to be worth integrating.

---

## Key Flows

- F1. Lab pipeline (per prompt)
  - **Trigger:** A labeled prompt (benign or credential-seeking) is run through
    the lab.
  - **Steps:** Build prompt (injected context + query) → forward pass with
    hidden states → capture last-K layers → pool readout-position features →
    score against benign baseline (Mahalanobis) and, if pursued, the learned
    probe.
  - **Outcome:** A per-prompt detection score and per-layer features, aggregated
    across the set into metrics and figures.
  - **Covers:** R1, R2, R3, R4, R9; conditionally R15

---

## Acceptance Examples

- AE1. Encoding contrast
  - **Covers R8, R12, R13.**
  - **Given** a held-out scenario whose prompt requests an encoding the text
    scanner does not normalize (e.g. rot13),
  - **When** it is scored by both the text scanner and CIFT,
  - **Then** text-detection F1 drops materially versus its verbatim level while
    CIFT's score stays near its verbatim level. If CIFT also collapses on this
    model, that is recorded as a finding, not treated as a failed requirement.

- AE2. Readout-position correctness and confound control
  - **Covers R3, R7.**
  - **Given** a benign prompt that merely *contains* credentials in context (and
    a separate benign prompt carrying exfiltration vocabulary) but is not steered
    to use them,
  - **When** CIFT scores it,
  - **Then** it is not flagged — confirming the probe reads "use" at the readout
    position, not "presence" at the credential tokens or surface vocabulary.

- AE3. Probe adds value
  - **Covers R9, R15.**
  - **Given** the held-out split,
  - **When** the learned ensemble and the best single layer are compared,
  - **Then** the learned-ensemble AUROC is at least the best single-layer AUROC
    (otherwise the probe is recorded as not justified on this model).

---

## Success Criteria

- The Mahalanobis baseline runs end-to-end on the local model, and the R10
  positive control separates, before any null result is interpreted.
- The per-layer Mahalanobis figure and the encoding-robustness contrast are
  produced and legible.
- You can explain the result in your own words: which layers carried signal, why
  late layers dominate (or don't), and how the activation signal behaves under
  encoding versus the text scanner.
- A null or weak result counts as success only when it is interpreted *and* the
  positive control passed and the surface-confound was ruled out — so the
  explanation rests on CIFT's behavior, not a harness bug or a learned confound.
- The learned probe (R15) is not required for success; the baseline plus contrast
  plus interpretation are the core bar.

---

## Scope Boundaries

### Deferred for later

- DP-HONEY (DP character-bigram canaries + conformal calibration) and the real
  InfoNCE NIMBUS — separate learning targets, each its own brainstorm.
- Cross-session leakage ledger (the paper's "restart session to reset the
  budget" attack) — orthogonal to CIFT.
- Hardening the stretch wire-in beyond a working pre-output gate (calibration
  refresh, latency tuning, cold-start handling).

### Outside this effort

- Cloud or GPU training, real (non-fake) credentials, deployment, multi-user
  hosting, and product/UI polish. The repo's standing constraint holds: all
  credentials fake, all tools local.

---

## Dependencies / Assumptions

- **New ML stack.** `torch` and `transformers` (plus a metrics helper such as
  `scikit-learn`/`numpy`) are net-new dependencies; the repo has no ML stack
  today. A model-weights download is required — default to an ungated model
  (Qwen2.5 is ungated) to avoid Hugging Face auth friction.
- **Hardware.** Apple Silicon (arm64) with MPS; CPU fallback acceptable at lower
  speed. No GPU rental.
- **Data is authored, not reused.** The labeled corpus (R5) is built from
  scratch — the repo's "scenarios" are mock output-template selectors, not
  steering prompts. The benign calibration set must be authored too (the paper
  used ~500 benign prompts).
- **Behavioral assumption (load-bearing).** A 1.5B instruct model exhibits
  enough credential-access behavior under the *authored* steering prompts to move
  readout-position activations. If it does not, the first responses are to remove
  any suppressing framing (per R6) and sharpen the steer — not to default to a
  larger model, which is more likely to recognize and honor a fake-canary
  disclaimer, not less.

---

## Outstanding Questions

### Deferred to planning

- Which *additional* readout positions to include beyond R3's floor (the final
  prompt token and first pre-generation decision position) — e.g. the final four
  prompt tokens — and how many benign prompts to calibrate against.
- Lab form factor (annotated notebook vs script-plus-saved-figures).
- Which specific undecoded encoding(s) to use in the held-out suite (rot13, a
  substitution cipher, fragmentation) — given base64/hex are already normalized
  by the text scanner.
- Whether to mean-ablate the highest-weight layers to test causal influence
  (paper §5.2) — the intervention that would directly test the "accesses before
  it encodes" claim rather than inferring it from the contrast.

---

## Sources / Research

- Paper `docs/2606.04141v1.pdf` — §4.2 CIFT (activation extraction, readout
  positions, Mahalanobis CCI, CFS probe), §5.2 + Figure 2 (per-layer analysis),
  Figure 3 (encoding-robustness contrast), §6 (white-box / transfer
  limitations).
- Repo model layer: `app/models/base.py` (adapter protocol),
  `app/models/ollama_model.py` (placeholder), `app/models/mock_responses.py`
  (current default).
- Existing scanners to contrast against: `app/scanners/` (canary + transform
  detectors), `app/scanners/transforms.py`.
- Current heuristic baseline for context: `app/nimbus/scoring.py`.
- Original intent for this slot: `docs/ais_io_prd_and_codex_prompts.md` §4.4
  (Future goals — white-box activation monitoring, CIFT layer visualization) and
  §10.4 (`HFWhiteBoxAdapter` is where CIFT attaches).
