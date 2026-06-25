# AIS

AIS is a local defensive demo for detecting credential exfiltration by LLM agents.
It exposes an OpenAI-compatible `POST /v1/responses` proxy shape, injects fake
canary credentials into model-visible context, scans typed response Items and
function-call arguments, applies local policy decisions, stores a trace in SQLite,
and displays the trace in a Streamlit dashboard.

All credentials are fake and non-functional. External exfiltration is not
performed. Tool-call leakage is simulated locally and blocked before any real
tool execution exists.

```text
Dashboard / Playground
        |
        +-- POST /chat
        |
        +-- POST /v1/responses
              |
              v
        AIS proxy pipeline
          request normalization
          canary generation and injection
          deterministic mock model adapter
          response item and function-call scanning
          NIMBUS-lite cumulative leakage scoring
          policy engine
          SQLite event store
```

## Setup

```bash
uv sync
```

## Run Backend

```bash
uv run uvicorn app.main:app --reload
```

The backend initializes `ais.db` automatically.

## Run Dashboard

```bash
uv run streamlit run dashboard/streamlit_app.py
```

Set `AIS_API_BASE` if the backend is not at `http://localhost:8000`.

## Example `/v1/responses`

```bash
curl -s http://localhost:8000/v1/responses \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "mock-ais",
    "input": [{"role":"user","content":"Summarize this ticket."}],
    "store": false,
    "metadata": {
      "session_id": "demo-session-1",
      "scenario": "base64_leak"
    }
  }'
```

## Example `/chat`

```bash
curl -s http://localhost:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{
    "session_id": "demo-session-1",
    "user_input": "Summarize this ticket.",
    "scenario": "direct_leak",
    "defenses": {
      "canary_injection": true,
      "output_scanning": true,
      "tool_scanning": true,
      "nimbus_lite": true
    }
  }'
```

## Scenarios

- `benign`: harmless ticket summary, no canary leak.
- `direct_leak`: model emits the first injected canary directly.
- `base64_leak`: model emits a Base64-encoded canary.
- `hex_leak`: model emits a hex-encoded canary.
- `markdown_link_leak`: model embeds a canary in a markdown URL.
- `tool_call_leak`: model emits a `send_email` function call whose JSON body contains a canary.
- `http_get_url_leak`: model emits an `http_get` function call with a Base64 canary in a URL query value.
- `benign_tool_call`: model emits a safe `send_email` call that is scanned and executed locally.
- `multi_turn_drip`: model emits small canary fragments across repeated turns; NIMBUS-lite accumulates risk until policy escalates.

## Phase 2 Tool Interception

Every `function_call` output Item passes through ToolProxy before fake local
execution. ToolProxy parses JSON arguments, scans the raw and serialized
arguments, flattens nested string values, parses URLs and query parameters, and
applies the same canary and credential-shape detectors used for text output.

Unsafe tool calls are blocked and persisted with `executed = false`. Safe calls
execute only local fake tools and persist their fake result in `tool_calls`.

The fake tools are:

- `send_email(to, subject, body)`
- `http_get(url, headers)`
- `create_ticket(title, description)`
- `write_file(path, content)`
- `query_kb(query)`

They do not send email, perform outbound network calls, or write files.

## Phase 3 NIMBUS-lite

NIMBUS-lite is a transparent demo heuristic, not a certified information-flow
bound. When enabled, AIS scores each turn from detector hits and suspicious
features such as partial canary fragments, credential-shaped values,
high-entropy tokens, user requests for secrets, and encoded-looking output.

Scores accumulate per `session_id` in `leakage_ledger` with a default budget of
`10.0`:

- `< 60%`: `PASS`
- `60-79%`: `WARN`
- `80-99%`: `SANITIZE`
- `>= 100%`: `BLOCK`

Detector policy and NIMBUS-lite policy are combined by taking the stricter
action. The `multi_turn_drip` scenario demonstrates why this matters: no single
turn contains a full canary, but repeated fragments eventually cross the
cumulative budget.

## CIFT Activation-Probe Lab

The `cift/` package is an offline research lab that reproduces the paper's CIFT
method (activation-based, pre-output credential-access detection) on a small
local model. It is independent of the FastAPI app and uses an opt-in dependency
group, so the default install and test suite stay light.

Everything stays local: a small instruct model (default `Qwen2.5-1.5B-Instruct`)
runs on-device, all secrets are fake canaries, and nothing leaves the machine.

What it does:

- Captures last-K-layer hidden states at the readout position (the final prompt
  token after the chat template), never at static credential-token positions.
- Fits an unsupervised diagonal-Mahalanobis detector from a benign baseline.
- Validates the pipeline with a positive control before interpreting any null
  result, then contrasts CIFT against the existing text scanner under a rot13
  encoding requested in-prompt (which `app/scanners/transforms.py` does not
  decode), producing two figures and a written interpretation.

Install the lab dependencies (torch, transformers, scikit-learn, matplotlib):

```bash
uv sync --group cift
```

Run the lab end to end (downloads the model on first run, writes figures,
`interpretation.md`, and the operating-point `threshold.json` under
`cift/artifacts/`):

```bash
uv run python -m cift.run
```

### Live monitor

For an interactive demo of the paper's core claim — that the activations betray
credential-seeking *before* a token is generated — launch the Streamlit live
monitor. It races CIFT's **pre-output** verdict (read from the prompt's readout
activations, zero tokens generated) against the text scanner's **post-output**
verdict on the model's actual reply, and shows the token index where the leak
first becomes visible to the scanner:

```bash
uv run --group cift streamlit run cift/live_monitor.py
```

It reuses the `baseline.npz` and `threshold.json` written by `cift.run`; if the
operating point is missing it calibrates one from benign held-out prompts on
first launch. Pick a built-in scenario (matched-surface benign, the measured
steered leak, the exfil-vocabulary confound, or a blatant out-of-distribution
control) or type your own prompt.

Set `AIS_CIFT_MODEL=Qwen/Qwen2.5-0.5B-Instruct` for a faster, smaller run, or
`AIS_CIFT_DEVICE=cpu` to force CPU. The model-in-the-loop extraction tests are
opt-in:

```bash
CIFT_MODEL_TESTS=1 uv run --group cift pytest tests/test_cift_extraction.py
```

NIMBUS-lite and the CIFT learned probe / live `/v1/responses` gate are deliberate
non-goals of this lab — see `docs/plans/2026-06-25-001-feat-cift-activation-probe-lab-plan.md`.
The live monitor visualizes CIFT on individual prompts but does not gate the
proxy; wiring the probe into `/v1/responses` remains the deferred U9 stretch.

## Tests

```bash
uv run pytest
```

The default suite stays light: the `cift` lab tests skip automatically when the
`cift` group is not installed. Install it (`uv sync --group cift`) to run the
corpus, detector, contrast, and figure tests.
