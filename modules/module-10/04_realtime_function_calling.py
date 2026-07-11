"""Domain 10 - Task 10.4: Realtime Function Calling

CONCEPTS:
  1. Tools in session config — same JSON schema as Chat API
  2. response.function_call_arguments.delta — streaming args
  3. conversation.item.create — send function result
  4. response.create — trigger next response

Mnemonic: TFDC — Tools, Function_call_delta, Create_result, Respond

Run:
  uv run python 04_realtime_function_calling.py
"""

from __future__ import annotations

import json
from collections.abc import Generator
from dataclasses import dataclass, field

from shared.mock import get_client, is_mock

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

MODEL = "gpt-4o"
REALTIME_MODEL = "gpt-4o-realtime-preview"
NL = chr(10)

# ---------------------------------------------------------------------------
# Tool definition helpers
# ---------------------------------------------------------------------------


def build_weather_tool() -> dict[str, object]:
    """Return a Realtime API tool definition for get_weather.

    NOTE: Realtime API uses flat schema (name/description/parameters at top
    level), NOT nested under a 'function' key like Chat Completions API.
    """
    return {
        "type": "function",
        "name": "get_weather",
        "description": "Get current weather for a given city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "The city to get weather for.",
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "Temperature unit.",
                },
            },
            "required": ["city"],
        },
    }


def build_session_with_tools(tools: list[dict[str, object]]) -> dict[str, object]:
    """Return a session.update payload that registers tools."""
    return {
        "type": "session.update",
        "session": {
            "tools": tools,
            "tool_choice": "auto",
        },
    }


# ---------------------------------------------------------------------------
# Server-to-client event dataclasses
# ---------------------------------------------------------------------------


@dataclass
class FunctionCallArgumentsDeltaEvent:
    """Server streams partial JSON arguments for a function call."""

    type: str = "response.function_call_arguments.delta"
    event_id: str = "event_mock040"
    response_id: str = "resp_mock001"
    item_id: str = "item_mock001"
    call_id: str = "call_mock001"
    delta: str = ""


@dataclass
class FunctionCallArgumentsDoneEvent:
    """Server signals function call arguments are complete."""

    type: str = "response.function_call_arguments.done"
    event_id: str = "event_mock041"
    response_id: str = "resp_mock001"
    item_id: str = "item_mock001"
    call_id: str = "call_mock001"
    name: str = ""
    arguments: str = "{}"


@dataclass
class ResponseDoneEvent:
    """Server signals full response is complete."""

    type: str = "response.done"
    event_id: str = "event_mock050"
    response_id: str = "resp_mock001"
    status: str = "completed"


# ---------------------------------------------------------------------------
# Client-to-server event dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ConversationItemCreateEvent:
    """Client sends a function result back into the conversation."""

    type: str = "conversation.item.create"
    item_type: str = "function_call_output"
    call_id: str = "call_mock001"
    output: str = "{}"

    def to_dict(self) -> dict[str, object]:
        return {
            "type": self.type,
            "item": {
                "type": self.item_type,
                "call_id": self.call_id,
                "output": self.output,
            },
        }


@dataclass
class ResponseCreateEvent:
    """Client requests model to generate a follow-up response."""

    type: str = "response.create"
    modalities: list[str] = field(default_factory=lambda: ["text"])

    def to_dict(self) -> dict[str, object]:
        return {
            "type": self.type,
            "response": {"modalities": self.modalities},
        }


# ---------------------------------------------------------------------------
# Tool executor
# ---------------------------------------------------------------------------


def execute_get_weather(city: str, unit: str = "celsius") -> dict[str, object]:
    """Mock implementation of get_weather tool."""
    mock_data: dict[str, dict[str, object]] = {
        "london": {"temperature": 15, "condition": "cloudy", "humidity": 78},
        "paris": {"temperature": 18, "condition": "sunny", "humidity": 60},
        "tokyo": {"temperature": 25, "condition": "partly cloudy", "humidity": 70},
    }
    data = mock_data.get(city.lower(), {"temperature": 20, "condition": "unknown", "humidity": 65})
    if unit == "fahrenheit":
        data["temperature"] = int(data["temperature"]) * 9 // 5 + 32  # type: ignore[operator]
    return {"city": city, "unit": unit, **data}


# ---------------------------------------------------------------------------
# Mock event streams
# ---------------------------------------------------------------------------


def mock_function_call_stream(
    function_name: str = "get_weather",
    arguments: dict[str, object] | None = None,
) -> Generator[
    FunctionCallArgumentsDeltaEvent | FunctionCallArgumentsDoneEvent | ResponseDoneEvent,
    None,
    None,
]:
    """Simulate server streaming function call arguments."""
    if arguments is None:
        arguments = {"city": "London", "unit": "celsius"}

    args_str = json.dumps(arguments)
    call_id = "call_mock001"

    # Stream arguments character by character (simulating real streaming)
    chunk_size = 5
    for i in range(0, len(args_str), chunk_size):
        yield FunctionCallArgumentsDeltaEvent(
            event_id=f"event_arg_delta_{i:03d}",
            call_id=call_id,
            delta=args_str[i : i + chunk_size],
        )

    yield FunctionCallArgumentsDoneEvent(
        event_id="event_arg_done",
        call_id=call_id,
        name=function_name,
        arguments=args_str,
    )
    yield ResponseDoneEvent(event_id="event_resp_done")


# ---------------------------------------------------------------------------
# Demo functions
# ---------------------------------------------------------------------------


def demo_tool_configuration() -> None:
    """DEMO 1: Show tool configuration in realtime session."""
    tool = build_weather_tool()
    session_payload = build_session_with_tools([tool])

    print("Tool definition (Realtime API flat schema):")
    print(f"  type       : {tool['type']}")
    print(f"  name       : {tool['name']}")
    print(f"  description: {tool['description']}")
    print(f"  parameters : {json.dumps(tool['parameters'], indent=4)}")
    print()
    print("session.update payload (abridged):")
    print(f"  type         : {session_payload['type']}")
    session_obj = session_payload["session"]
    print(f"  tool_choice  : {session_obj['tool_choice']}")  # type: ignore[index]
    print(f"  tools count  : {len(session_obj['tools'])}")  # type: ignore[index]


def demo_anti_pattern_chat_schema() -> None:
    """ANTI-PATTERN 1: Using Chat Completions nested tool schema in Realtime API."""
    wrong_tool = {
        "type": "function",
        "function": {  # WRONG — 'function' key not used in Realtime API
            "name": "get_weather",
            "description": "Get weather.",
            "parameters": {"type": "object", "properties": {}},
        },
    }
    print("ANTI-PATTERN: Chat Completions tool schema (WRONG for Realtime API):")
    print(f"  {wrong_tool}")
    print("  -> Realtime API uses flat schema: name/description/parameters at top level")


def demo_function_call_streaming() -> None:
    """DEMO 2: Simulate receiving streaming function call arguments."""
    func_name = "get_weather"
    args = {"city": "Paris", "unit": "celsius"}

    print(f"Streaming function call: {func_name}")
    accumulated_args = ""
    final_args: dict[str, object] = {}

    for event in mock_function_call_stream(func_name, args):
        if isinstance(event, FunctionCallArgumentsDeltaEvent):
            accumulated_args += event.delta
        elif isinstance(event, FunctionCallArgumentsDoneEvent):
            final_args = json.loads(event.arguments)
            print(f"  Function name    : {event.name}")
            print(f"  Accumulated args : {accumulated_args!r}")
            print(f"  Parsed args      : {final_args}")
        elif isinstance(event, ResponseDoneEvent):
            print(f"  Response status  : {event.status}")


def demo_tool_result_roundtrip() -> None:
    """DEMO 3: Execute tool and send result back via conversation.item.create."""
    call_id = "call_mock001"
    func_args = {"city": "Tokyo", "unit": "fahrenheit"}

    print("Executing tool: get_weather")
    result = execute_get_weather(**func_args)  # type: ignore[arg-type]
    result_json = json.dumps(result)
    print(f"  Tool result: {result}")

    # Build conversation.item.create to send result back
    item_create = ConversationItemCreateEvent(
        call_id=call_id,
        output=result_json,
    )
    print()
    print("Sending function result back (conversation.item.create):")
    payload = item_create.to_dict()
    print(f"  type      : {payload['type']}")
    item = payload["item"]
    print(f"  item.type : {item['type']}")  # type: ignore[index]
    print(f"  call_id   : {item['call_id']}")  # type: ignore[index]
    print(f"  output    : {item['output']}")  # type: ignore[index]

    # Trigger next response
    resp_create = ResponseCreateEvent(modalities=["text"])
    print()
    print("Triggering follow-up response (response.create):")
    print(f"  {resp_create.to_dict()}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _client = get_client()
    _mock_note = "(mock mode)" if is_mock() else "(live mode)"
    sep = "=" * 60

    print(sep)
    print(f"Domain 10 - Task 10.4: Realtime Function Calling {_mock_note}")
    print(sep)

    print(NL + "DEMO 1: Tool Configuration in Session")
    print("-" * 40)
    demo_tool_configuration()

    print(NL + "ANTI-PATTERN 1: Chat Completions Nested Schema")
    print("-" * 40)
    demo_anti_pattern_chat_schema()

    print(NL + "DEMO 2: Streaming Function Call Arguments")
    print("-" * 40)
    demo_function_call_streaming()

    print(NL + "DEMO 3: Tool Result Roundtrip")
    print("-" * 40)
    demo_tool_result_roundtrip()

    print(NL + sep)
    print("KEY TAKEAWAYS:")
    print("  - Realtime tools use flat schema: name/description/parameters at top level (no 'function' key)")
    print("  - response.function_call_arguments.delta streams JSON args incrementally")
    print("  - After execution send result via conversation.item.create with type=function_call_output")
    print("  - response.create triggers the model's follow-up response after the tool result")
    print("  - call_id links the function_call_arguments events to the matching function_call_output")
    print(sep)
