# Project 1: Persistent CLI Chatbot

## What it does
A CLI chatbot with conversation history, tool use (calculator + time), and structured logging on every API call.

## How to run
```
uv run python main.py --demo
```

## With real API
```
OPENAI_API_KEY=sk-... uv run python main.py
```

## Architecture
- `Chatbot` class manages conversation history (trimmed to last 10 turns)
- `logged_api_call()` wraps every completion with structured JSON logging
- Tool loop: `finish_reason == "tool_calls"` → execute → append tool results → continue

## Key SDK patterns used
- `client.chat.completions.create()` with `tools=` parameter
- `finish_reason == "stop"` to terminate the agentic loop
- `extra_headers` for structured logging correlation
- History trimming to control token cost
