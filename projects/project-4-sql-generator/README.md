# Project 4: Natural Language to SQL Generator

## What it does
Converts natural language questions to SQL queries using schema-aware generation with a validation retry loop and mutation keyword blocking.

## How to run
```
uv run python main.py --demo
```

## With real API
```
OPENAI_API_KEY=sk-... uv run python main.py
```

## Architecture
- Schema injected into system prompt so the model knows table/column names
- `validate_sql()`: regex-based checks (must start with SELECT, no mutation keywords)
- Retry loop: invalid SQL triggers a correction message up to `MAX_RETRIES` times
- `SQLResult` Pydantic model wraps output with `is_valid` and `error` fields

## Key SDK patterns used
- Long system prompt with schema context
- Validation retry loop with error feedback in messages
- `response_format` NOT used — SQL is free text, not JSON
