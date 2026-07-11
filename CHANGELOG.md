# Changelog

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
