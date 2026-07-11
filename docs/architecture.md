# Architecture

## Repository Layout

```
finish-reason-stop/
├── pyproject.toml          # UV workspace root
├── shared/                 # Shared utility library (workspace member)
│   └── shared/
│       ├── mock.py         # MockClient, AsyncMockClient, get_client()
│       ├── retry.py        # exponential_backoff decorator
│       ├── tokens.py       # count_tokens, estimate_cost, format_cost
│       ├── logging.py      # structured_log, APICallLogger
│       ├── config.py       # Config.from_env()
│       └── types.py        # Pydantic models, TypedDicts
├── modules/module-{1..12}/ # Learning modules (12 workspace members)
├── exercises/exercise-*-*/ # Synthesis exercises (6 workspace members)
├── projects/project-*-*/   # Mini-projects (6 workspace members)
└── .claude/                # Claude Code configuration
```

## UV Workspace

All directories under `modules/`, `exercises/`, `projects/`, and `shared/` are workspace members sharing a single `.venv` and `uv.lock`. Each member's `pyproject.toml` declares:

```toml
dependencies = ["shared"]
[tool.uv.sources]
shared = {workspace = true}
```

This enables `from shared.mock import get_client` in every script.

## Mock Mode Architecture

`shared/shared/mock.py` provides a complete duck-typed mock of the OpenAI SDK. Key design principles:

1. **Plain dataclasses** — never subclass openai's internal Pydantic models (they have `__init_subclass__` guards)
2. **Namespace classes** — `_ChatCompletionsNamespace`, `_ResponsesNamespace`, etc. mirror the real client's attribute hierarchy
3. **Duck typing** — code that calls `client.chat.completions.create()` works identically against `MockClient` or `openai.OpenAI()`

```
MockClient
├── responses          → _ResponsesNamespace.create() → MockResponse
├── chat.completions   → _ChatCompletionsNamespace.create() → MockChatCompletion
├── embeddings         → _EmbeddingsNamespace.create() → MockCreateEmbeddingResponse
├── images             → _ImagesNamespace.generate() → SimpleNamespace
├── audio              → _AudioNamespace (transcriptions, translations, speech)
├── files              → _FilesNamespace
├── batches            → _BatchesNamespace
├── fine_tuning.jobs   → _FineTuningNamespace
└── vector_stores      → _VectorStoresNamespace
```

## Script Structure

Every demo script follows an identical structure:

```
module-level docstring (CONCEPTS + Mnemonic + Run)
NL = chr(10)
MODEL = "gpt-4o"
imports (shared.mock first)
constants
dataclasses / Pydantic models
def demo_N() → None:
def anti_pattern_N() → None:
if __name__ == "__main__":
    sep = "=" * 60
    print header [MOCK|LIVE]
    demo calls
    KEY TAKEAWAYS
```

## Module Dependencies

```
shared  ←  all modules, exercises, projects
Module 1 (Responses API basics)
Module 2 (Tool Use) — builds on Module 1
Module 3 (Structured Output) — complements Module 2
Module 4 (Streaming/Async) — complements Module 2
Module 5 (Embeddings) — standalone
Module 6 (Audio) — standalone
Module 7 (Images/Vision) — complements Module 3
Module 8 (Files/RAG) — builds on Module 5
Module 9 (Batch/Fine-tune) — builds on Module 5
Module 10 (Realtime) — standalone
Module 11 (Error Handling) — cross-cutting
Module 12 (Production) — cross-cutting

Exercises: combine 2-3 modules
Projects: end-to-end, use all relevant modules
```
