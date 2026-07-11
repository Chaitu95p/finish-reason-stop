"""Domain 2 - Task 2.2: Tool Orchestration Loop (Canonical Example)

CONCEPTS:
  1. Agentic loop — call API → detect tool_calls → execute → call again
  2. finish_reason == "tool_calls" — not done; must execute tools and continue
  3. finish_reason == "stop" — loop terminates naturally
  4. Tool results — role="tool", tool_call_id=..., content=string

Mnemonic: LACE — Loop, Append_assistant, Call_tool, Execute

Run:
  uv run python 02_tool_orchestration_loop.py
"""

NL = chr(10)
MODEL = "gpt-4o"

import json

from shared.mock import MockToolCall, get_client, is_mock

# ---------------------------------------------------------------------------
# Mock tool implementation
# ---------------------------------------------------------------------------

def get_weather(city: str) -> str:
    """Pure mock: return weather data as JSON string."""
    data = {"city": city, "temp_c": 21, "condition": "partly cloudy", "humidity": 60}
    return json.dumps(data)


def execute_tool(tool_call: MockToolCall) -> str:
    """Dispatch a tool call to the right Python function."""
    name = tool_call.function.name
    try:
        args: dict = json.loads(tool_call.function.arguments)
    except json.JSONDecodeError:
        return json.dumps({"error": "Invalid JSON in arguments"})

    if name == "get_weather":
        return get_weather(args.get("city", "unknown"))
    return json.dumps({"error": f"Unknown tool: {name!r}"})


# ---------------------------------------------------------------------------
# Tool schema
# ---------------------------------------------------------------------------

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"}
            },
            "required": ["city"],
            "additionalProperties": False,
        },
    },
}


def demo_tool_loop() -> None:
    """DEMO 1: Complete agentic loop — the canonical pattern."""
    client = get_client()
    messages: list[dict] = [
        {"role": "user", "content": "What is the weather in Tokyo?"}
    ]
    tools = [WEATHER_TOOL]
    max_iters = 10

    print("  Starting agentic loop...")
    for iteration in range(max_iters):
        completion = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=tools,
        )
        choice = completion.choices[0]
        finish_reason = choice.finish_reason
        print(f"  Iteration {iteration + 1}: finish_reason={finish_reason!r}")

        if finish_reason == "stop":
            print(f"  Final answer: {choice.message.content!r}")
            break

        if finish_reason == "tool_calls":
            # Append the assistant's tool-call message first (required!)
            messages.append({
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": tc.type,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in choice.message.tool_calls
                ],
            })

            # Execute each tool and append results
            for tool_call in choice.message.tool_calls:
                result = execute_tool(tool_call)
                print(f"    Tool {tool_call.function.name!r} → {result}")
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
                })
        else:
            print(f"  Unexpected finish_reason: {finish_reason!r}")
            break


def demo_forced_tool_call() -> None:
    """DEMO 2: Force a tool call via tool_choice='required' to see the tool_calls path."""
    client = get_client()
    messages: list[dict] = [{"role": "user", "content": "Check Tokyo weather"}]

    # tool_choice="required" forces MockClient to return finish_reason="tool_calls"
    completion = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=[WEATHER_TOOL],
        tool_choice="required",
    )
    reason = completion.choices[0].finish_reason
    tool_calls = completion.choices[0].message.tool_calls
    print(f"  finish_reason: {reason!r}")
    if tool_calls:
        tc = tool_calls[0]
        print(f"  Tool called: {tc.function.name!r}")
        print(f"  Arguments:   {tc.function.arguments!r}")
        result = execute_tool(tc)
        print(f"  Result:      {result!r}")


def anti_pattern_fixed_cap() -> None:
    """ANTI-PATTERN 1: Fixed iteration cap as the primary stop condition.

    Using 'for i in range(5)' and breaking on the last iteration
    means the loop can terminate MID-TASK if the model needs more steps.
    finish_reason is the CORRECT signal; iteration count is a safety backstop.
    """
    client = get_client()
    messages: list[dict] = [{"role": "user", "content": "Analyze markets"}]
    tools = [WEATHER_TOOL]

    # WRONG — breaks at 5 even if the model is mid-tool-call
    for i in range(5):
        completion = client.chat.completions.create(model=MODEL, messages=messages, tools=tools)
        if i == 4:
            # silently truncates the conversation
            print("  [ANTI-PATTERN] Broke after 5 iters regardless of finish_reason")
            break

    print("  CORRECT: use while True: ... if finish_reason == 'stop': break")
    print("  Use max_iters as a SAFETY backstop, not the primary stop condition")


if __name__ == "__main__":
    sep = "=" * 60
    mode = "MOCK" if is_mock() else "LIVE"
    print(f"{NL}{sep}{NL}Domain 2 - Task 2.2: Tool Orchestration Loop [{mode}]{NL}{sep}")

    print(f"{NL}--- DEMO 1: Full Agentic Loop ---")
    demo_tool_loop()

    print(f"{NL}--- DEMO 2: Forced Tool Call Path ---")
    demo_forced_tool_call()

    print(f"{NL}--- ANTI-PATTERN 1: Fixed Iteration Cap ---")
    anti_pattern_fixed_cap()

    print(f"{NL}KEY TAKEAWAYS:")
    print("  1. finish_reason == 'tool_calls' → execute tools and call API again")
    print("  2. finish_reason == 'stop' → loop terminates; use the final message")
    print("  3. ALWAYS append the assistant message BEFORE the tool result messages")
    print("  4. Iteration cap is a safety backstop, not the primary termination signal")
