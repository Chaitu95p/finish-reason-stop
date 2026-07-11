# finish-reason-stop — OpenAI Python SDK Learning Repo

A structured learning repository for the OpenAI Python SDK. Every script runs in **mock mode** without an API key.

## Quick Commands

```bash
# Run any script in mock mode (no API key needed)
uv run python modules/module-1/01_responses_api_basics.py

# Run with real API
OPENAI_API_KEY=sk-... uv run python modules/module-1/01_responses_api_basics.py

# Sync workspace dependencies
uv sync --all-packages

# Lint all scripts
uv run ruff check shared/ modules/ exercises/ projects/
```

## Module Table

| # | Name | Primary API | Scripts |
|---|------|-------------|---------|
| 1 | Responses API & Chat Completions | `client.responses.create()` | 5 |
| 2 | Function Calling & Tool Use | `tools=`, `finish_reason="tool_calls"` | 6 |
| 3 | Structured Outputs & JSON Mode | `response_format`, Pydantic | 6 |
| 4 | Streaming & Async | `stream=True`, `AsyncOpenAI` | 6 |
| 5 | Embeddings & Semantic Search | `client.embeddings.create()` | 6 |
| 6 | Audio APIs | `audio.transcriptions`, `audio.speech` | 5 |
| 7 | Images & Vision | `images.generate()`, vision messages | 6 |
| 8 | Files, Vector Stores & RAG | `client.files`, `client.vector_stores` | 5 |
| 9 | Batch API & Fine-Tuning | `client.batches`, `client.fine_tuning` | 6 |
| 10 | Realtime API | WebSocket sessions | 5 |
| 11 | Error Handling & Reliability | `openai.APIError` hierarchy | 6 |
| 12 | Production, Testing & Security | Logging, cost, security | 7 |

## Script Conventions (enforced by .claude/rules/demo-scripts.md)

```python
NL = chr(10)        # NEVER use \n inline in f-strings
MODEL = "gpt-4o"   # always at module level

from shared.mock import get_client, is_mock  # NEVER import openai.OpenAI directly

def demo_something() -> None:  # type hints on ALL parameters and return types
    """DEMO 1: What this demonstrates."""
    ...
```

## Responses API Loop Pattern

```python
response = client.responses.create(
    model="gpt-4o",
    input="Hello",
    instructions="You are helpful.",
)
if response.status == "completed":
    print(response.output_text)

# Stateful multi-turn
response2 = client.responses.create(
    model="gpt-4o",
    input="Follow-up question",
    previous_response_id=response.id,
)
```

## Tool Use Loop Pattern (Chat Completions)

```python
while True:
    response = client.chat.completions.create(model=MODEL, messages=messages, tools=TOOLS)
    choice = response.choices[0]
    messages.append({"role": "assistant", "content": choice.message.content, "tool_calls": [...]})

    if choice.finish_reason == "stop":
        break
    if choice.finish_reason == "tool_calls":
        for tc in choice.message.tool_calls:
            result = execute_tool(tc.function.name, json.loads(tc.function.arguments))
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
```

## Mock Mode

`get_client()` returns `MockClient` when `OPENAI_API_KEY` is absent. All API surfaces are simulated:

```python
from shared.mock import get_client, is_mock, get_async_client

client = get_client()          # MockClient or openai.OpenAI()
is_mock()                      # True when no API key
aclient = await get_async_client()  # AsyncMockClient or AsyncOpenAI()
```

## shared/ Library Reference

| Module | Purpose |
|--------|---------|
| `shared.mock` | `get_client()`, `is_mock()`, MockClient, AsyncMockClient |
| `shared.retry` | `@exponential_backoff(max_retries, base_delay, max_delay)` decorator |
| `shared.tokens` | `count_tokens(text)`, `estimate_cost(pt, ct, model)`, `format_cost(usd)` |
| `shared.logging` | `structured_log(event, **kwargs)`, `new_request_id()`, `APICallLogger` |
| `shared.config` | `Config.from_env()` |
| `shared.types` | `MessageDict`, `SentimentResult`, `SummaryResult` Pydantic models |

## .claude/ Config

- `settings.json` — allow/deny lists for Bash; PreToolUse/PostToolUse hooks
- `hooks/pre-tool-use.sh` — blocks dangerous commands (rm -rf, force push, pip install)
- `hooks/post-tool-use.sh` — logs all tool calls to `.claude/tool-usage.log`
- `commands/` — `/run-module`, `/run-script`, `/review-script`, `/new-demo`, `/run-project`
- `skills/` — `explore-module`, `generate-quiz`, `check-coverage`
- `rules/demo-scripts.md` — auto-injected for modules/ and exercises/
- `rules/projects.md` — auto-injected for projects/

## @import Examples

@import modules/module-1/01_responses_api_basics.py
@import modules/module-2/02_tool_orchestration_loop.py
