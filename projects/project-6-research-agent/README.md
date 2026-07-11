# Project 6: Research Agent

## What it does
An agentic researcher that uses web_search, fetch_page, and take_note tools in a proper agentic loop, then synthesizes findings into a structured research report.

## How to run
```
uv run python main.py --demo
uv run python main.py --query "What is quantum computing?"
```

## With real API
```
OPENAI_API_KEY=sk-... uv run python main.py --query "History of the internet"
```

## Architecture
- Three tools: `web_search`, `fetch_page`, `take_note` (all simulated in mock mode)
- Agentic loop: agent decides when it has enough notes to synthesize the report
- `_notes` list accumulates findings across tool calls
- Agent terminates with `finish_reason == "stop"` after synthesizing

## Key SDK patterns used
- Multi-tool agentic loop with `finish_reason == "tool_calls"`
- Safe tool dispatch: all tools wrapped in try/except, return JSON errors
- System prompt guides agent reasoning strategy (search → fetch → note → synthesize)
