# finish-reason-stop

A structured learning repository for the OpenAI Python SDK. Covers every major API surface through 90+ runnable scripts, 6 exercises, and 6 mini-projects. Every script runs without an API key via mock mode.

## Quickstart

```bash
# Install dependencies (Python 3.12+, uv required)
uv sync --all-packages

# Run any script — no API key needed
uv run python modules/module-1/01_responses_api_basics.py

# Run with real API
OPENAI_API_KEY=sk-... uv run python modules/module-1/01_responses_api_basics.py
```

## Module Table

| Module | Topic | Key APIs |
|--------|-------|----------|
| 1 | Responses API & Chat Completions | `responses.create()`, `chat.completions.create()` |
| 2 | Function Calling & Tool Use | `tools=`, `finish_reason="tool_calls"` |
| 3 | Structured Outputs & JSON Mode | `response_format`, Pydantic, `beta.chat.completions.parse()` |
| 4 | Streaming & Async (7 scripts) | `stream=True`, `AsyncOpenAI`, `asyncio.gather()`, Responses API streaming |
| 5 | Embeddings & Semantic Search | `embeddings.create()`, cosine similarity, RAG |
| 6 | Audio APIs | `audio.transcriptions.create()`, `audio.speech.create()` |
| 7 | Images & Vision | `images.generate()`, vision content blocks |
| 8 | Files, Vector Stores & RAG | `files.create()`, `vector_stores.create()` |
| 9 | Batch API & Fine-Tuning | `batches.create()`, `fine_tuning.jobs.create()` |
| 10 | Realtime API | WebSocket sessions, audio streaming |
| 11 | Error Handling & Reliability | Error taxonomy, retry, circuit breaker |
| 12 | Production, Testing & Security (9 scripts) | Logging, cost, security, testing, reasoning models, prompt caching |

## Projects

| Project | Description |
|---------|-------------|
| 1 | CLI chatbot with tool use and conversation history |
| 2 | RAG document Q&A with citations |
| 3 | Resume analyzer with structured extraction |
| 4 | Natural language to SQL with validation retry |
| 5 | Meeting summarizer (audio → structured report) |
| 6 | Research agent with multi-tool web search |

## Learning Path

1. Start with **Module 1** — understand Responses API vs Chat Completions
2. **Module 2** — master the tool-use agentic loop (critical pattern)
3. **Module 3** — structured outputs with Pydantic
4. **Module 4** — async for concurrent requests
5. **Module 5** — embeddings and RAG
6. Modules 6-10 — domain-specific APIs (audio, vision, batch, etc.)
7. Modules 11-12 — production reliability and testing
8. **Exercises** — synthesize across modules
9. **Projects** — end-to-end applications

## Repository Structure

```
finish-reason-stop/
├── shared/           # Shared utilities: MockClient, retry, tokens, logging
├── modules/          # 12 learning modules, 90+ scripts
├── exercises/        # 6 synthesis exercises
├── projects/         # 6 mini-projects (end-to-end applications)
├── docs/             # Architecture, SDK reference, migration guide
├── .claude/          # Claude Code config: hooks, commands, skills, rules
├── CLAUDE.md         # Claude Code instructions for this repo
└── cheatsheet.md     # SDK quick reference
```

## Mock Mode

All scripts use `from shared.mock import get_client, is_mock`. When `OPENAI_API_KEY` is absent, `get_client()` returns a `MockClient` with duck-typed responses that match the real OpenAI SDK interface.

```python
from shared.mock import get_client, is_mock

client = get_client()
print("Mock mode:", is_mock())
resp = client.responses.create(model="gpt-4o", input="Hello")
print(resp.output_text)  # works in mock mode
```
