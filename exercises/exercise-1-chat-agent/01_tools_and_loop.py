"""Exercise 1 - Task 1.1: Tools and Agentic Loop

GOAL: Build a working tool-use agent that calls get_weather and calculator
tools in a proper finish_reason loop.

SKILLS PRACTICED:
  - Function definition with JSON schema
  - finish_reason == "tool_calls" loop
  - Executing tool results and appending to messages

Run:
  uv run python 01_tools_and_loop.py
"""

NL = chr(10)
MODEL = "gpt-4o"

import json

from shared.mock import get_client, is_mock

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"},
                },
                "required": ["city"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate a simple arithmetic expression.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Math expression, e.g. '2 + 3 * 4'"},
                },
                "required": ["expression"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    },
]


def get_weather(city: str) -> str:
    """Simulated weather tool."""
    return json.dumps({"city": city, "temp_c": 22, "condition": "sunny"})


def calculator(expression: str) -> str:
    """Safe arithmetic evaluator."""
    try:
        allowed = set("0123456789+-*/()., ")
        if not all(c in allowed for c in expression):
            return json.dumps({"error": "invalid characters"})
        result = eval(expression, {"__builtins__": {}})  # noqa: S307
        return json.dumps({"result": result})
    except Exception as e:
        return json.dumps({"error": str(e)})


def execute_tool(name: str, args: dict) -> str:
    """Dispatch to the appropriate tool function."""
    if name == "get_weather":
        return get_weather(args["city"])
    if name == "calculator":
        return calculator(args["expression"])
    return json.dumps({"error": f"unknown tool: {name}"})


def run_agent(user_message: str) -> str:
    """Run the tool-use agent loop until finish_reason == 'stop'."""
    client = get_client()
    messages: list[dict] = [{"role": "user", "content": user_message}]

    while True:
        response = client.chat.completions.create(  # type: ignore[attr-defined]
            model=MODEL,
            messages=messages,
            tools=TOOLS,
        )
        choice = response.choices[0]
        finish_reason = choice.finish_reason
        msg = choice.message

        # Append assistant message (may have tool_calls)
        assistant_msg: dict = {"role": "assistant", "content": msg.content}
        tool_calls = getattr(msg, "tool_calls", None) or []
        if tool_calls:
            assistant_msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in tool_calls
            ]
        messages.append(assistant_msg)

        if finish_reason == "stop":
            return msg.content or ""

        if finish_reason == "tool_calls":
            for tc in tool_calls:
                args = json.loads(tc.function.arguments or "{}")
                result = execute_tool(tc.function.name, args)
                print(f"  [tool] {tc.function.name}({args}) → {result}")
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })
            continue

        break

    return ""


if __name__ == "__main__":
    sep = "=" * 60
    mode = "MOCK" if is_mock() else "LIVE"
    print(f"{NL}{sep}{NL}Exercise 1 - Task 1.1: Tools and Loop [{mode}]{NL}{sep}")

    queries = [
        "What's the weather in Tokyo and what is 15 * 7?",
        "What is (100 + 200) / 3?",
    ]
    for query in queries:
        print(f"{NL}  Query: {query!r}")
        answer = run_agent(query)
        print(f"  Answer: {answer!r}")

    print(f"{NL}KEY TAKEAWAYS:")
    print("  1. Loop until finish_reason == 'stop' — not a fixed iteration count")
    print("  2. Every tool_call must get a tool result before the next API call")
    print("  3. Append both assistant message (with tool_calls) AND tool results to messages")
