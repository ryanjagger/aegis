from __future__ import annotations

import os
import sys
from pathlib import Path

# `streamlit run dashboard/streamlit_app.py` puts the script's directory
# (dashboard/) on sys.path, not the repo root, and this project is not
# pip-installed (the tests rely on pytest's pythonpath="."). The AIS Walkthrough
# page imports the repo-root `dphoney` package (and, via it, `app`), so put the
# repo root on the path ourselves before that import runs.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import json  # noqa: E402
from typing import Any  # noqa: E402

import requests  # noqa: E402
import streamlit as st  # noqa: E402

API_BASE = os.getenv("AIS_API_BASE", "http://localhost:8000").rstrip("/")


def api_get(path: str) -> Any:
    response = requests.get(f"{API_BASE}{path}", timeout=10)
    response.raise_for_status()
    return response.json()


def api_post(path: str, payload: dict[str, Any]) -> Any:
    response = requests.post(f"{API_BASE}{path}", json=payload, timeout=20)
    response.raise_for_status()
    return response.json()


def show_table(path: str) -> list[dict[str, Any]]:
    try:
        rows = api_get(path)
    except requests.RequestException as exc:
        st.error(f"Backend request failed: {exc}")
        return []
    if rows:
        st.dataframe(rows, use_container_width=True)
    else:
        st.info("No rows yet.")
    return rows


def live_requests() -> None:
    st.header("Live Requests")
    rows = show_table("/requests")
    if rows:
        latest = rows[0]
        st.subheader("Latest")
        st.json(latest)


def response_trace() -> None:
    st.header("Response Trace")
    responses = show_table("/responses")
    if not responses:
        return
    response_ids = [row["id"] for row in responses]
    selected = st.selectbox("Response", response_ids)
    trace = api_get(f"/responses/{selected}")
    st.subheader("Request")
    st.json(trace.get("raw_request_json"))
    st.subheader("Normalized Input Items")
    st.json(trace.get("normalized_input_json"))
    st.subheader("Injected Context")
    st.code(trace.get("injected_context") or "", language="text")
    st.subheader("Raw Response")
    st.json(trace.get("raw_response_json"))
    st.subheader("Final Response")
    st.json(trace.get("final_response_json"))
    st.subheader("Detector Hits")
    st.dataframe(trace.get("detector_events", []), use_container_width=True)
    st.subheader("Canaries")
    st.dataframe(trace.get("canaries", []), use_container_width=True)
    st.subheader("Tool Calls")
    st.dataframe(trace.get("tool_calls", []), use_container_width=True)
    st.subheader("Leakage Ledger")
    st.dataframe(trace.get("leakage_ledger", []), use_container_width=True)
    st.subheader("Event Timeline")
    st.dataframe(trace.get("events", []), use_container_width=True)


def canary_registry() -> None:
    st.header("Canary Registry")
    show_table("/canaries")


def attack_playground() -> None:
    st.header("Attack Playground")
    with st.form("attack_playground_form"):
        route = st.selectbox("Route", ["/chat", "/v1/responses"])
        scenario = st.selectbox(
            "Scenario",
            [
                "benign",
                "direct_leak",
                "base64_leak",
                "hex_leak",
                "markdown_link_leak",
                "tool_call_leak",
                "http_get_url_leak",
                "benign_tool_call",
                "multi_turn_drip",
            ],
        )
        model_adapter = st.selectbox("Model adapter", ["mock", "openai", "ollama"])
        session_id = st.text_input("Session ID", "demo-session-1")
        user_input = st.text_area(
            "User input",
            "Summarize this support ticket and create an internal note.",
            height=100,
        )
        col1, col2, col3, col4 = st.columns(4)
        canary_injection = col1.checkbox("Canaries", value=True)
        output_scanning = col2.checkbox("Output scan", value=True)
        tool_scanning = col3.checkbox("Tool scan", value=True)
        nimbus_lite = col4.checkbox("NIMBUS", value=False)
        canary_source = st.selectbox(
            "Canary source", ["template", "dp"],
            help="template = deterministic format; dp = DP-HONEY model (harder to filter).",
        )
        submitted = st.form_submit_button("Run")

    if not submitted:
        return

    payload = {
        "route": route,
        "session_id": session_id,
        "user_input": user_input,
        "scenario": scenario,
        "model_adapter": model_adapter,
        "defenses": {
            "canary_injection": canary_injection,
            "canary_source": canary_source,
            "output_scanning": output_scanning,
            "tool_scanning": tool_scanning,
            "nimbus_lite": nimbus_lite,
        },
    }
    try:
        result = api_post("/playground/run", payload)
    except requests.RequestException as exc:
        st.error(f"Playground request failed: {exc}")
        return
    st.subheader("Response")
    st.json(result)
    st.code(json.dumps(result, indent=2), language="json")


def tool_calls() -> None:
    st.header("Tool Calls")
    rows = show_table("/tool-calls")
    if not rows:
        return

    call_ids = [row["call_id"] for row in rows]
    selected_call_id = st.selectbox("Tool call", call_ids)
    selected = next(row for row in rows if row["call_id"] == selected_call_id)
    st.subheader("Arguments")
    st.json(selected.get("arguments_json"))
    st.subheader("Result")
    st.json(selected.get("result_json"))
    st.subheader("Related Detector Hits")
    try:
        detector_events = api_get("/detector-events")
    except requests.RequestException as exc:
        st.error(f"Detector event request failed: {exc}")
        return
    related = [
        event
        for event in detector_events
        if event.get("request_id") == selected.get("request_id")
        and "function_call.arguments" in event.get("surface", "")
    ]
    if related:
        st.dataframe(related, use_container_width=True)
    else:
        st.info("No function-call detector hits for this tool call.")


def leakage_ledger() -> None:
    st.header("Leakage Ledger")
    rows = show_table("/leakage-ledger")
    if rows:
        st.line_chart(rows, x="created_at", y="score_total")
    else:
        st.caption("Enable NIMBUS in the Attack Playground to populate this ledger.")


def ais_walkthrough() -> None:
    from dphoney import live_logic
    from dphoney.artifacts import SEPARABILITY_PNG, artifacts_dir

    st.header("AIS Walkthrough")
    st.caption("The full pipeline in one pass: DP-HONEY → CIFT → text backstop → NIMBUS.")

    # One run action drives every downstream station; the result is held in
    # session state so it survives reruns (slider moves, etc.).
    col_run, col_scenario = st.columns([1, 2])
    scenario = col_scenario.selectbox(
        "Scenario",
        ["benign", "direct_leak", "base64_leak", "tool_call_leak", "multi_turn_drip"],
    )
    if col_run.button("Inject DP canary + run", type="primary"):
        payload = {
            "route": "/chat",
            "session_id": "walkthrough",
            "user_input": "Summarize this support ticket and create an internal note.",
            "scenario": scenario,
            "model_adapter": "mock",
            "defenses": {
                "canary_injection": True,
                "canary_source": "dp",
                "output_scanning": True,
                "tool_scanning": True,
                "nimbus_lite": True,
            },
        }
        try:
            with st.spinner("Running the live pipeline…"):
                st.session_state["walkthrough_result"] = api_post("/playground/run", payload)
        except requests.RequestException as exc:
            st.error(f"Pipeline request failed: {exc}")
    result = st.session_state.get("walkthrough_result")

    # 1 · DP-HONEY ----------------------------------------------------------------
    st.markdown("### 1 · DP-HONEY — calibrated honeytokens")
    figure = artifacts_dir() / SEPARABILITY_PNG
    if figure.exists():
        st.image(str(figure), caption=live_logic.separability_caption(True))
    else:
        st.info(live_logic.separability_caption(False))
    with st.expander("Canary-accounting calculator — Pr(detect) = k/(m+k)·(1−β)"):
        k = st.slider("planted canaries k", 0, 50, 5)
        m = st.slider("real credentials m", 0, 50, 5)
        beta = st.slider("detector miss rate β", 0.0, 1.0, 0.1)
        st.metric("Pr(detect)", f"{live_logic.accounting(k, m, beta):.3f}")

    # 2 · CIFT --------------------------------------------------------------------
    st.markdown("### 2 · CIFT — pre-output activation gate")
    st.caption("Opt-in white-box station; launch it separately (loads the model):")
    st.code(live_logic.cift_launch_command())

    # 3 · Text backstop -----------------------------------------------------------
    st.markdown("### 3 · Text backstop — canary scanner")
    if result:
        st.json(result)
    else:
        st.info(live_logic.prerun_message("text backstop"))

    # 4 · NIMBUS ------------------------------------------------------------------
    st.markdown("### 4 · NIMBUS — cumulative leakage")
    st.warning(live_logic.nimbus_label())
    if result:
        try:
            rows = api_get("/leakage-ledger")
        except requests.RequestException as exc:
            st.error(f"Ledger request failed: {exc}")
            rows = []
        if rows:
            st.line_chart(rows, x="created_at", y="score_total")
        else:
            st.caption("Ledger empty for this run.")
    else:
        st.info(live_logic.prerun_message("NIMBUS"))


def main() -> None:
    st.set_page_config(page_title="AIS", layout="wide")
    st.title("AIS")
    st.caption("Local defensive demo for credential exfiltration in agent outputs and tool calls.")
    st.sidebar.header("Backend")
    st.sidebar.code(API_BASE)
    page = st.sidebar.radio(
        "Page",
        [
            "AIS Walkthrough",
            "Live Requests",
            "Response Trace",
            "Canary Registry",
            "Attack Playground",
            "Tool Calls",
            "Leakage Ledger",
        ],
    )
    pages = {
        "AIS Walkthrough": ais_walkthrough,
        "Live Requests": live_requests,
        "Response Trace": response_trace,
        "Canary Registry": canary_registry,
        "Attack Playground": attack_playground,
        "Tool Calls": tool_calls,
        "Leakage Ledger": leakage_ledger,
    }
    pages[page]()


if __name__ == "__main__":
    main()
