"""Domain 2 - Task 2.3: Tool Choice Control

CONCEPTS:
  1. tool_choice="auto" — model decides whether to call a tool (default)
  2. tool_choice="required" — model must use at least one tool
  3. tool_choice={"type":"function","function":{"name":...}} — force specific tool
  4. tool_choice="none" — disable tool use; pure text response

Mnemonic: ARFN — Auto, Required, Forced_name, None

Run:
  uv run python 03_tool_choice_control.py
"""

NL = chr(10)
MODEL = "gpt-4o"

from shared.mock import get_client, is_mock

TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
            "additionalProperties": False,
        },
    },
}

CALC_TOOL = {
    "type": "function",
    "function": {
        "name": "calculate",
        "description": "Perform arithmetic.",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "Math expression"}
            },
            "required": ["expression"],
            "additionalProperties": False,
        },
    },
}


def demo_auto() -> None:
    """DEMO 1: tool_choice='auto' — model decides."""
    client = get_client()
    completion = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "What is the weather in Paris?"}],
        tools=[TOOL],
        tool_choice="auto",
    )
    reason = completion.choices[0].finish_reason
    print(f"  tool_choice='auto' → finish_reason={reason!r}")
    print("  Use auto when: you want the model to decide based on context")


def demo_required() -> None:
    """DEMO 2: tool_choice='required' — model must call a tool."""
    client = get_client()
    # MockClient returns tool_calls when tool_choice="required"
    completion = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "Hello!"}],
        tools=[TOOL, CALC_TOOL],
        tool_choice="required",
    )
    reason = completion.choices[0].finish_reason
    calls = completion.choices[0].message.tool_calls
    print(f"  tool_choice='required' → finish_reason={reason!r}")
    if calls:
        print(f"  Called: {calls[0].function.name!r}")
    print("  Use required when: you always need data collection before answering")


def demo_forced_specific() -> None:
    """DEMO 3: tool_choice=forced function — use exactly this tool."""
    client = get_client()
    forced = {"type": "function", "function": {"name": "get_weather"}}
    completion = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "What should I wear today?"}],
        tools=[TOOL, CALC_TOOL],
        tool_choice=forced,
    )
    reason = completion.choices[0].finish_reason
    print(f"  tool_choice=forced → finish_reason={reason!r}")
    print("  Use forced when: you need a specific tool regardless of query")


def demo_none() -> None:
    """DEMO 4: tool_choice='none' — no tool use; pure text."""
    client = get_client()
    completion = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "Tell me about the weather."}],
        tools=[TOOL],
        tool_choice="none",
    )
    reason = completion.choices[0].finish_reason
    content = completion.choices[0].message.content
    print(f"  tool_choice='none' → finish_reason={reason!r}")
    print(f"  Content: {content!r}")
    print("  Use none when: you want a general explanation, not live data")


if __name__ == "__main__":
    sep = "=" * 60
    mode = "MOCK" if is_mock() else "LIVE"
    print(f"{NL}{sep}{NL}Domain 2 - Task 2.3: Tool Choice Control [{mode}]{NL}{sep}")

    print(f"{NL}--- DEMO 1: auto ---")
    demo_auto()

    print(f"{NL}--- DEMO 2: required ---")
    demo_required()

    print(f"{NL}--- DEMO 3: forced specific tool ---")
    demo_forced_specific()

    print(f"{NL}--- DEMO 4: none ---")
    demo_none()

    print(f"{NL}KEY TAKEAWAYS:")
    print("  1. 'auto' is default — model judges when a tool is needed")
    print("  2. 'required' guarantees a tool call — use for data-collection flows")
    print("  3. Forced by name — use when you know exactly which tool is needed")
    print("  4. 'none' disables tools — use for explanation/summary steps in an agent")
