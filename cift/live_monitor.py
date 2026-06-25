"""CIFT Live Monitor — the paper's pre-output claim, made visible (Streamlit).

    uv run --group cift streamlit run cift/live_monitor.py

The whole point of CIFT (arXiv:2606.04141) is that the readout-position
activations betray credential-seeking *before* the model emits a single token.
A text scanner can only react to output that already exists. This page races the
two: a prompt's pre-output Mahalanobis verdict (0 tokens generated) on the left,
the model's actual generated reply and the text scanner's verdict on the right,
with the token index where the leak finally becomes visible to the scanner.

Honesty notes baked into the UI (this lab's whole theme):
- The gauge measures distance from the *matched-surface benign* distribution the
  baseline was fit on. On a 1.5B model the steered-vs-benign separation is real
  but modest (AUROC ~0.68) and scales with model size — the UI says so rather
  than dressing it up.
- The "blatant leak" scenario is labelled an out-of-distribution control: it
  scores high partly because it is short/OOD, not only because it seeks a secret.
  The matched-surface steered-leak scenario is the honest measured contrast.
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import streamlit as st

from app.canaries.generator import first_canary_like_value
from app.config import get_settings
from app.scanners.canary_scanner import CanaryScanner
from cift.corpus import blatant_control_prompts, build_corpus
from cift.detector import (
    load_baseline,
    load_operating_point,
    operating_point_from_benign_scores,
    per_layer_scores,
    save_operating_point,
    score,
)
from cift.extraction import extract_features, extract_many, generate_text, get_model
from cift.live_logic import gauge_fraction, layer_labels, verdict_is_attack

ARTIFACTS_SUBPATH = ("cift", "artifacts")
CALIBRATION_N = 24


@st.cache_resource(show_spinner=False)
def load_lab():
    """Load the fitted baseline and the model singleton (process-lifetime)."""

    settings = get_settings()
    artifacts = Path(settings.data_dir).joinpath(*ARTIFACTS_SUBPATH)
    baseline_path = artifacts / "baseline.npz"
    if not baseline_path.exists():
        raise FileNotFoundError(baseline_path)
    baseline = load_baseline(baseline_path)
    (tokenizer, model), device = get_model()
    return settings, artifacts, baseline, tokenizer, model, device


@st.cache_resource(show_spinner=False)
def load_operating(_baseline, artifacts_str: str, n_calib: int):
    """Load the persisted operating point, or calibrate one from benign held-out.

    Unsupervised: a benign-quantile threshold, so no attack labels are needed. The
    result is cached and persisted to ``threshold.json`` so this only runs once.
    """

    path = Path(artifacts_str) / "threshold.json"
    if path.exists():
        return load_operating_point(path), False
    corpus = build_corpus(seed=get_settings().cift_seed)
    benign_texts = [p.text for p in corpus.benign_heldout[:n_calib]]
    benign_scores = score(_baseline, extract_many(benign_texts))
    op = operating_point_from_benign_scores(benign_scores)
    save_operating_point(op, path)
    return op, True


@st.cache_data(show_spinner=False)
def load_scenarios():
    """Authored demo prompts, each with the secret value (if any) to scan for."""

    corpus = build_corpus(seed=get_settings().cift_seed)
    benign = corpus.benign_heldout[0]
    attack = corpus.attack_heldout[0]
    confound = corpus.confound_control
    blatant_benign, blatant_attack = blatant_control_prompts(n=1)
    return {
        "Benign — matched surface (expect green)": {
            "text": benign.text,
            "secret": benign.canary.value if benign.canary else None,
            "note": "Carries the secret in context but does not steer toward using it.",
        },
        "Steered leak — matched surface (the measured case)": {
            "text": attack.text,
            "secret": attack.canary.value if attack.canary else None,
            "note": "Same surface as benign; only the steer differs (the AUROC ~0.68 contrast).",
        },
        "Confound — exfil vocabulary, no steer (expect green)": {
            "text": confound.text,
            "secret": confound.canary.value if confound.canary else None,
            "note": "Exfil/phishing words but asks only for a summary — CIFT should not flag it.",
        },
        "Blatant leak — obvious control (out-of-distribution)": {
            "text": blatant_attack[0],
            "secret": first_canary_like_value(blatant_attack[0]),
            "note": "Explicit 'output this secret' — reliably emits it, but short/OOD.",
        },
    }


def render_pre_output(baseline, model, text: str, op):
    feats = extract_features(text)  # [K, hidden]
    per_layer = per_layer_scores(baseline, feats)[0]  # [K]
    total = float(per_layer.sum())
    is_attack = verdict_is_attack(total, op.threshold)

    st.subheader("Pre-output verdict")
    st.caption("0 tokens generated — read from the prompt's readout-position activations")
    st.progress(gauge_fraction(total, op.threshold, op.benign_std))
    if is_attack:
        st.error(f"🔴 CREDENTIAL-SEEKING — score {total:.1f} > threshold {op.threshold:.1f}")
    else:
        st.success(f"🟢 benign-like — score {total:.1f} ≤ threshold {op.threshold:.1f}")

    labels = layer_labels(model.config.num_hidden_layers, len(per_layer))
    chart = pd.DataFrame({"Mahalanobis distance": per_layer}, index=labels)
    chart.index.name = "monitored layer"
    st.bar_chart(chart)
    return total


def render_output_and_scanner(
    tokenizer, text: str, secret: str | None, max_new: int, animate: bool
):
    st.subheader("Model output (greedy)")
    out_box = st.empty()
    with st.spinner("Generating…"):
        output = generate_text(text, max_new_tokens=max_new)

    leak_token: int | None = None
    if animate and output:
        ids = tokenizer(output, add_special_tokens=False).input_ids
        for i in range(len(ids)):
            shown = tokenizer.decode(ids[: i + 1], skip_special_tokens=True)
            out_box.code(shown or "…")
            if secret and leak_token is None and secret in shown:
                leak_token = i + 1
            time.sleep(0.02)
        out_box.code(output or "(empty)")
    else:
        out_box.code(output or "(empty)")
        if secret and secret in output:
            prefix = output[: output.find(secret) + len(secret)]
            leak_token = len(tokenizer(prefix, add_special_tokens=False).input_ids)

    hits = []
    if secret:
        hits = CanaryScanner([{"id": "watched", "value": secret}]).scan_text(output, "output_text")

    if hits:
        where = f"token {leak_token}" if leak_token else "in output"
        st.error(f"🔴 Text scanner: leak detected ({hits[0].detector}) {where}")
    elif secret:
        st.success("🟢 Text scanner: secret not present in output")
    else:
        st.info("No secret registered — the pre-output verdict above is independent of this.")
    return hits, leak_token


def main() -> None:
    st.set_page_config(page_title="CIFT Live Monitor", layout="wide")
    st.title("CIFT Live Monitor")
    st.caption(
        "Activation probing flags credential-seeking **before** a token is generated; "
        "the text scanner can only react **after**. Watch the gap."
    )

    try:
        settings, artifacts, baseline, tokenizer, model, _ = load_lab()
    except FileNotFoundError as exc:
        st.error(
            f"No fitted baseline at `{exc}`.\n\n"
            "Run the lab first to produce it:\n\n"
            "```\nuv run --group cift python -m cift.run\n```"
        )
        st.stop()

    with st.spinner("Calibrating operating point (first run only)…"):
        op, freshly_calibrated = load_operating(baseline, str(artifacts), CALIBRATION_N)
    if freshly_calibrated:
        st.toast("Calibrated an operating point from benign held-out prompts.")

    st.sidebar.header("Settings")
    animate = st.sidebar.checkbox("Animate output (token-by-token reveal)", value=True)
    max_new = st.sidebar.slider("Max new tokens", 16, 160, 96, 16)
    st.sidebar.metric("Operating threshold", f"{op.threshold:.1f}")
    st.sidebar.caption(
        f"~{int(round(op.target_fpr * 100))}% benign false-positive rate · "
        f"model `{settings.cift_model_name}`"
    )
    st.sidebar.caption(
        "Gauge = distance from the matched-surface benign cloud. On 1.5B the steered-vs-benign "
        "separation is real but modest (AUROC ~0.68) and scales with model size."
    )

    scenarios = load_scenarios()
    names = list(scenarios.keys())
    choice = st.selectbox("Scenario", [*names, "Custom…"])

    if choice == "Custom…":
        text = st.text_area("Prompt", value="", height=180, placeholder="Type any prompt…")
        manual = st.text_input("Secret to watch for (optional)", value="")
        secret = manual.strip() or (first_canary_like_value(text) if text else None)
    else:
        st.caption(scenarios[choice]["note"])
        text = st.text_area("Prompt", value=scenarios[choice]["text"], height=180)
        secret = scenarios[choice]["secret"]

    if not st.button("Run", type="primary") or not text.strip():
        return

    left, right = st.columns(2)
    with left:
        render_pre_output(baseline, model, text, op)
    with right:
        hits, leak_token = render_output_and_scanner(
            tokenizer, text, secret, max_new, animate
        )

    st.markdown("---")
    if hits:
        st.markdown(
            f"**⏱ CIFT produced its verdict at token 0 — the text scanner first saw the leak at "
            f"token {leak_token if leak_token else '?'}.** That window is the pre-output advantage."
        )
    else:
        st.markdown(
            "**⏱ CIFT produced a verdict before any token was generated** — no output had to exist "
            "for the activation probe to decide."
        )


main()
