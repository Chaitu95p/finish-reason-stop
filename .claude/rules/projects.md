# Project Rules

**Applies to:** `projects/project-*/`

These rules are automatically injected when working on mini-project files.

## Required Files

Every project must have:
- `main.py` — entry point, runnable end-to-end
- `README.md` — what it does, how to run, architecture notes
- `pyproject.toml` — workspace member config

## main.py Requirements

1. **Mock mode mandatory:** Must run without `OPENAI_API_KEY`
   ```python
   from shared.mock import get_client, is_mock
   ```

2. **Clear CLI output:** Print what the project is doing at each step

3. **Error handling:** Graceful degradation, never crash on API errors

4. **Type hints:** All functions fully typed

5. **Entry point:**
   ```python
   if __name__ == "__main__":
       main()
   ```

## README.md Requirements

```markdown
# Project N: <Project Name>

## What it does
[1-2 sentences]

## How to run
uv run python main.py

## With real API
OPENAI_API_KEY=sk-... uv run python main.py

## Architecture
[Brief description of components]

## Key SDK patterns used
- Pattern 1
- Pattern 2
```

## pyproject.toml Requirements

```toml
[project]
name = "project-N-name"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = ["shared"]

[tool.uv.sources]
shared = {workspace = true}
```
