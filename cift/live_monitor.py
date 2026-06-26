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

import sys
from pathlib import Path

# `streamlit run cift/live_monitor.py` puts the script's directory (cift/) on
# sys.path, not the repo root, and this project is not pip-installed (the tests
# rely on pytest's pythonpath="."). So `import app`/`import cift` would fail —
# put the repo root on the path ourselves before importing them.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import time  # noqa: E402

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from app.canaries.generator import first_canary_like_value  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.scanners.canary_scanner import CanaryScanner  # noqa: E402
from cift.corpus import (  # noqa: E402
    blatant_control_prompts,
    build_corpus,
    indirect_injection_prompt,
    intent_without_leak_prompt,
)
from cift.detector import (  # noqa: E402
    load_baseline,
    load_operating_point,
    operating_point_from_benign_scores,
    per_layer_scores,
    save_operating_point,
    score,
)
from cift.extraction import (  # noqa: E402
    extract_features,
    extract_many,
    fingerprint,
    generate_text,
    get_model,
)
from cift.live_logic import (  # noqa: E402
    fingerprint_mismatch,
    footer_message,
    gauge_fraction,
    layer_labels,
    verdict_is_attack,
)

ARTIFACTS_SUBPATH = ("cift", "artifacts")
# Calibrate the operating threshold on a solid benign sample; a small sample gives
# a noisy 95th-percentile estimate (the threshold every verdict depends on).
CALIBRATION_N = 60


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
    indirect = indirect_injection_prompt()
    intent = intent_without_leak_prompt()
    encoded = corpus.encoding_suite[0]
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
        "Indirect prompt injection — steer hidden in the document": {
            "text": indirect.text,
            "secret": indirect.canary.value if indirect.canary else None,
            "note": (
                "Leak steer buried in the untrusted document (benign user instruction). "
                "Measured: the model leaks but CIFT often misses it — the buried steer "
                "reads benign at the readout position. A real blind spot, only the "
                "scanner catches it."
            ),
        },
        "Intent without leak — credential use, no verbatim dump": {
            "text": intent.text,
            "secret": intent.canary.value if intent.canary else None,
            "note": (
                "Steers credential use, not a dump. Measured: CIFT flags the intent "
                "while the model acts without emitting the secret — the scanner stays "
                "silent. The clearest case for pre-output detection."
            ),
        },
        "Encoding leak (rot13) — hard case": {
            "text": encoded.encoded.text,
            "secret": encoded.canary.value,
            "note": (
                "ROT13 request the small model can't actually perform (failed encoding). "
                "CIFT fires, but the score is partly driven by the unusual/OOD prompt, "
                "like the blatant control."
            ),
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
    return is_attack


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
        settings, artifacts, baseline, tokenizer, model, device = load_lab()
    except FileNotFoundError as exc:
        st.error(
            f"No fitted baseline at `{exc}`.\n\n"
            "Run the lab first to produce it:\n\n"
            "```\nuv run --group cift python -m cift.run\n```"
        )
        st.stop()

    # Refuse to score against a baseline fit with a different model/device — its
    # scores would be meaningless. A version drift only warns.
    fp_diff = fingerprint_mismatch(baseline.fingerprint or {}, fingerprint(device))
    critical = {k: v for k, v in fp_diff.items() if k in ("model", "device")}
    if critical:
        rows = "\n".join(
            f"- {k}: baseline `{s}` vs current `{c}`" for k, (s, c) in critical.items()
        )
        st.error(
            "The fitted baseline does not match the current model/device, so its scores "
            f"would be meaningless:\n\n{rows}\n\nRe-fit with the current settings:\n\n"
            "```\nuv run --group cift python -m cift.run\n```"
        )
        st.stop()
    if fp_diff:  # only non-critical fields differ (torch/transformers/dtype)
        drift = ", ".join(f"{k}: {s} → {c}" for k, (s, c) in fp_diff.items())
        st.warning(f"Baseline was fit with a different stack ({drift}); results may have drifted.")

    with st.spinner(f"Preparing operating point (calibrates on {CALIBRATION_N} benign prompts)…"):
        op, freshly_calibrated = load_operating(baseline, str(artifacts), CALIBRATION_N)
    if freshly_calibrated:
        st.toast(f"Calibrated from {CALIBRATION_N} benign held-out prompts.")

    st.sidebar.header("Settings")
    animate = st.sidebar.checkbox("Animate output (token-by-token reveal)", value=True)
    max_new = st.sidebar.slider("Max new tokens", 16, 160, 96, 16)
    st.sidebar.metric("Operating threshold", f"{op.threshold:.1f}")
    st.sidebar.caption(
        f"~{int(round(op.target_fpr * 100))}% benign false-positive rate · "
        f"model `{settings.cift_model_name}`"
    )
    if st.sidebar.button("Recalibrate operating point"):
        (artifacts / "threshold.json").unlink(missing_ok=True)
        load_operating.clear()
        st.rerun()
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
        cift_flagged = render_pre_output(baseline, model, text, op)
    with right:
        hits, leak_token = render_output_and_scanner(
            tokenizer, text, secret, max_new, animate
        )

    st.markdown("---")
    scanner_fired = bool(hits)
    message = footer_message(
        cift_flagged=cift_flagged,
        scanner_fired=scanner_fired,
        scannable=bool(secret),
        leak_token=leak_token,
    )
    # A genuine CIFT miss (scanner caught a real leak that CIFT read as benign)
    # gets a warning box, not a triumphant timer line.
    if scanner_fired and not cift_flagged and secret:
        st.warning(f"⚠ {message}")
    else:
        st.markdown(f"**⏱ {message}**")


main()
