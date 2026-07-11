# Changelog

## v0.2.0 — Mock Layer Expansion + New Scripts + Test Suite

### Added
- **Test suite** (`tests/`) — 32 pytest tests for `shared.mock`, `shared.retry`, `shared.tokens`, `shared.logging`; added `"tests"` to pytest `testpaths`
- **`08_reasoning_models.py`** (module-12) — o1/o3/o4 patterns: `reasoning_effort`, `max_completion_tokens`, reading `reasoning_tokens`, o1 system-message restriction
- **`09_prompt_caching.py`** (module-12) — prompt caching: `cached_tokens`, cache-friendly layout, savings estimation with `shared.tokens`
- **`07_responses_api_streaming.py`** (module-4) — Responses API streaming: `stream=True`, event types, delta accumulation, tool call events

### Fixed
- **Async streaming** — `_MockStreamContextManager` now implements `__aenter__`/`__aexit__`/`__aiter__` so `async with` / `async for` work correctly; removed workaround comment from `modules/module-4/03_async_client_basics.py`
- **sdk-reference.md** Mock Equivalents table — corrected `prompt_tokens` (10→20), `completion_tokens` (20→30), `output_text`, `audio_response.text`, and `image_response.data[0].url` to match actual mock values

### Enhanced `shared/shared/mock.py`
- `MockUsage` — added `prompt_tokens_details` (with `cached_tokens`) and `completion_tokens_details` (with `reasoning_tokens`)
- `MockBatch` — added `error_file_id: str | None`; `_BatchesNamespace.retrieve_failed()` returns a failed batch with `error_file_id="file-mock-error-001"`
- `client.models` — `list()` returns 4 model objects; `retrieve(model_id)` returns a `MockModel`
- `client.moderations` — `create(input)` returns `MockModerationResponse` with `results[0].flagged=False`
- `client.beta.realtime` — `sessions.create()` returns `MockRealtimeSession`; `connect()` returns `MockRealtimeConnection` async context manager yielding 4 `MockRealtimeEvent` objects
- Reasoning model detection — o1/o3/o4 models return `completion_tokens_details.reasoning_tokens=150`
- Prompt cache simulation — repeated calls with same prefix return `cached_tokens=500`
- Responses API streaming — `_ResponsesNamespace.create(stream=True)` returns `_MockResponsesStreamContextManager` yielding `MockResponseEvent` objects

### Documentation
- `cheatsheet.md` — added Reasoning Models, Prompt Caching, Responses API Streaming, and Moderations sections
- `README.md` — updated script count to 90+; Module 4 noted as 7 scripts, Module 12 as 9 scripts

## v0.1.0 — Initial Release

### Added
- 12 learning modules covering all major OpenAI API surfaces (~70 scripts)
- 6 synthesis exercises combining multiple API concepts
- 6 runnable mini-projects (chatbot, RAG, resume analyzer, SQL generator, meeting summarizer, research agent)
- `shared/` utility library: MockClient, AsyncMockClient, retry, tokens, logging, config, types
- Full mock mode — all scripts run without `OPENAI_API_KEY`
- `.claude/` config: settings.json, hooks, commands, skills, rules
- Documentation: CLAUDE.md, README.md, cheatsheet.md, docs/

### Script conventions established
- `NL = chr(10)` — no inline `\n` in f-strings
- `from shared.mock import get_client, is_mock` — never direct openai import
- Type hints on all function signatures
- `DEMO N:` / `ANTI-PATTERN N:` section structure
- `KEY TAKEAWAYS:` in every `__main__` block

### Workspace setup
- UV workspace with Python >= 3.12
- Shared `.venv` across all workspace members
- `ruff` configured for py312, line-length 88
- `pytest` + `pytest-asyncio` for testing
