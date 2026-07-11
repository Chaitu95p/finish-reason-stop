"""Exercise 1 - Task 1.2: Error Recovery in Agent Loop

GOAL: Build a resilient agent that handles tool errors gracefully
without crashing the loop. The tool error is returned as JSON
and the model decides whether to retry or give up.

SKILLS PRACTICED:
  - Safe tool execution with try/except
  - Returning structured errors as tool results (not raising)
  - Agent loop resilience

Run:
  uv run python 02_error_recovery.py
"""

NL = chr(10)
MODEL = "gpt-4o"

import json
from typing import Any

from shared.mock import get_client, is_mock

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "divide",
            "description": "Divide two numbers. Returns an error if divisor is zero.",
            "parameters": {
                "type": "object",
                "properties": {
                    "dividend": {"type": "number"},
                    "divisor": {"type": "number"},
                },
                "required": ["dividend", "divisor"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    },
]


def safe_divide(dividend: float, divisor: float) -> str:
    """Divide; return JSON error on division by zero instead of raising."""
    try:
        if divisor == 0:
            return json.dumps({"error": "division by zero — please provide a non-zero divisor"})
        return json.dumps({"result": dividend / divisor})
    except Exception as e:
        return json.dumps({"error": str(e)})


def safe_execute_tool(name: str, arguments_json: str) -> str:
    """Execute any tool, catching ALL exceptions and returning JSON errors."""
    try:
        args = json.loads(arguments_json or "{}")
    except json.JSONDecodeError:
        return json.dumps({"error": "invalid arguments JSON"})
    try:
        if name == "divide":
            return safe_divide(args.get("dividend", 0), args.get("divisor", 0))
        return json.dumps({"error": f"unknown tool: {name}"})
    except Exception as e:
        return json.dumps({"error": f"tool raised: {type(e).__name__}: {e}"})


def run_resilient_agent(user_message: str) -> str:
    """Agent loop with safe tool execution."""
    client = get_client()
    messages: list[dict[str, Any]] = [{"role": "user", "content": user_message}]

    for _turn in range(10):
        response = client.chat.completions.create(  # type: ignore[attr-defined]
            model=MODEL,
            messages=messages,
            tools=TOOLS,
        )
        choice = response.choices[0]
        msg = choice.message
        tool_calls = getattr(msg, "tool_calls", None) or []

        assistant_entry: dict[str, Any] = {"role": "assistant", "content": msg.content}
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
                result = safe_execute_tool(tc.function.name, tc.function.arguments)
                print(f"  [tool] {tc.function.name} → {result}")
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

    return "Max turns reached."


if __name__ == "__main__":
    sep = "=" * 60
    mode = "MOCK" if is_mock() else "LIVE"
    print(f"{NL}{sep}{NL}Exercise 1 - Task 1.2: Error Recovery [{mode}]{NL}{sep}")

    tests = [
        "What is 100 divided by 4?",
        "What is 50 divided by 0?",
    ]
    for query in tests:
        print(f"{NL}  Query: {query!r}")
        result = run_resilient_agent(query)
        print(f"  Answer: {result!r}")

    print(f"{NL}KEY TAKEAWAYS:")
    print("  1. Tool functions must never raise — always return JSON with an 'error' key")
    print("  2. The model reads the error and can decide to retry or explain to the user")
    print("  3. safe_execute_tool wraps ALL tools — protects the loop from unexpected exceptions")
