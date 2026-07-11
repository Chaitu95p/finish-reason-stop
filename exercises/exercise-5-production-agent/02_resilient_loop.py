"""Exercise 5 - Task 5.2: Resilient Agent Loop

GOAL: Build an agent loop that handles API errors gracefully:
retries on transient errors, falls back to a smaller model on 429,
and never crashes on tool errors.

SKILLS PRACTICED:
  - Retry on APIConnectionError / RateLimitError
  - Model fallback chain on persistent rate limiting
  - Safe tool execution (never raises)

Run:
  uv run python 02_resilient_loop.py
"""

NL = chr(10)
PRIMARY_MODEL = "gpt-4o"
FALLBACK_MODEL = "gpt-4o-mini"
MAX_RETRIES = 3

import json
import time

import openai
from shared.mock import get_client, is_mock

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "lookup",
            "description": "Look up a value by key.",
            "parameters": {
                "type": "object",
                "properties": {"key": {"type": "string"}},
                "required": ["key"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    },
]


def safe_lookup(key: str) -> str:
    """Tool that never raises — returns errors as JSON."""
    database = {"pi": "3.14159", "e": "2.71828", "phi": "1.61803"}
    if key in database:
        return json.dumps({"value": database[key]})
    return json.dumps({"error": f"key '{key}' not found"})


def resilient_api_call(client: object, messages: list[dict], model: str) -> object:
    """Call API with retry and model fallback."""
    current_model = model
    for attempt in range(MAX_RETRIES):
        try:
            return client.chat.completions.create(  # type: ignore[attr-defined]
                model=current_model,
                messages=messages,
                tools=TOOLS,
            )
        except (openai.RateLimitError, openai.APIConnectionError, openai.APITimeoutError) as e:
            if current_model == PRIMARY_MODEL:
                print(f"  [retry] {type(e).__name__} — falling back to {FALLBACK_MODEL}")
                current_model = FALLBACK_MODEL
            if attempt >= MAX_RETRIES - 1:
                raise
            time.sleep(0.01 * (2 ** attempt))
    raise RuntimeError("unreachable")


def run_resilient_agent(user_message: str) -> str:
    """Resilient agent loop with error recovery."""
    client = get_client()
    messages: list[dict] = [{"role": "user", "content": user_message}]

    for _turn in range(10):
        response = resilient_api_call(client, messages, PRIMARY_MODEL)
        choice = response.choices[0]  # type: ignore[union-attr]
        msg = choice.message
        tool_calls = getattr(msg, "tool_calls", None) or []

        assistant_entry: dict = {"role": "assistant", "content": msg.content}
        if tool_calls:
            assistant_entry["tool_calls"] = [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in tool_calls
            ]
        messages.append(assistant_entry)

        if choice.finish_reason == "stop":
            return msg.content or ""

        if choice.finish_reason == "tool_calls":
            for tc in tool_calls:
                args = json.loads(tc.function.arguments or "{}")
                result = safe_lookup(**args)
                print(f"  [tool] lookup({args}) → {result}")
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

    return "Max turns reached."


if __name__ == "__main__":
    sep = "=" * 60
    mode = "MOCK" if is_mock() else "LIVE"
    print(f"{NL}{sep}{NL}Exercise 5 - Task 5.2: Resilient Loop [{mode}]{NL}{sep}")

    queries = [
        "What is the value of pi and e?",
        "What is the value of an unknown_key?",
    ]
    for query in queries:
        print(f"{NL}  Query: {query!r}")
        answer = run_resilient_agent(query)
        print(f"  Answer: {answer!r}")

    print(f"{NL}KEY TAKEAWAYS:")
    print("  1. Retry transient errors (429, connection) before giving up")
    print("  2. Fall back to smaller model on persistent rate limits")
    print("  3. Safe tool execution: return JSON errors, never raise")
