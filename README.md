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

## Tests

```bash
uv run pytest
```
