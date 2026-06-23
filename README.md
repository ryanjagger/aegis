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
          policy engine
          SQLite event store
```

## Setup

```bash
uv sync
```

## Run Backend

```bash
uvicorn app.main:app --reload
```

The backend initializes `ais.db` automatically.

## Run Dashboard

```bash
streamlit run dashboard/streamlit_app.py
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
      "nimbus_lite": false
    }
  }'
```

## Scenarios

- `benign`: harmless ticket summary, no canary leak.
- `direct_leak`: model emits the first injected canary directly.
- `base64_leak`: model emits a Base64-encoded canary.
- `hex_leak`: model emits a hex-encoded canary.
- `markdown_link_leak`: model embeds a canary in a markdown URL.
- `tool_call_leak`: model emits a `function_call` whose JSON arguments contain a canary.

## Tests

```bash
pytest
```

NIMBUS-lite and fake tool execution are reserved for later phases. The current
implementation includes schema/endpoints and placeholders for those areas, plus
function-call argument scanning and blocked tool-call trace persistence.
