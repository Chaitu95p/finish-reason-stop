"""Domain 4 - Task 4.7: Responses API Streaming

CONCEPTS:
  1. client.responses.create(stream=True) — SSE event iterator
  2. Event types — response.created, response.content_part.delta, response.completed
  3. Accumulating delta text across events
  4. How Responses API streaming differs from Chat streaming (event-based vs chunk-based)
  5. response.output_text on the final completed event

Mnemonic: SEADT — Stream, Events, Accumulate, Delta, Text

Run:
  uv run python 07_responses_api_streaming.py
"""

from __future__ import annotations

from shared.mock import MockResponseEvent, get_client, is_mock

NL = chr(10)
MODEL = "gpt-4o"


def demo_basic_streaming_events() -> None:
    """DEMO 1: Basic streaming with event type inspection."""
    client = get_client()
    print("Streaming Responses API events:")
    with client.responses.create(
        model=MODEL,
        input="Explain what a hash map is in one sentence.",
        stream=True,
    ) as stream:
        for event in stream:
            print(f"  event.type = {event.type!r}  event_id={event.event_id}")


def demo_accumulate_text_deltas() -> None:
    """DEMO 2: Accumulating text deltas into a full response string."""
    client = get_client()
    accumulated = ""
    final_text = ""

    with client.responses.create(
        model=MODEL,
        input="Name two benefits of streaming.",
        stream=True,
    ) as stream:
        for event in stream:
            if event.type == "response.content_part.delta":
                accumulated += event.delta
            elif event.type == "response.completed":
                final_text = event.output_text or accumulated

    print(f"Accumulated via deltas : {accumulated!r}")
    print(f"Final output_text      : {final_text!r}")


def demo_tool_call_events() -> None:
    """DEMO 3: Handling tool_call events in a streaming Responses API loop.

    In a real streaming Responses API call with tools, you would see events like:
      - response.output_item.added (type=function_call)
      - response.function_call_arguments.delta
      - response.function_call_arguments.done
      - response.output_item.done

    The mock stream yields simplified events. In production:
      1. Collect all function_call argument deltas into a buffer
      2. On response.function_call_arguments.done, parse the buffer as JSON
      3. Execute the tool and feed the result back via input=[{type: "tool_result", ...}]
    """
    print("Tool call streaming event flow (production pattern):")
    print()
    print("  for event in stream:")
    print('    if event.type == "response.output_item.added":')
    print('        if event.item["type"] == "function_call":')
    print('            current_call = event.item')
    print('    elif event.type == "response.function_call_arguments.delta":')
    print('        arg_buffer += event.delta')
    print('    elif event.type == "response.function_call_arguments.done":')
    print('        args = json.loads(arg_buffer)')
    print('        result = execute_tool(current_call["name"], args)')
    print('    elif event.type == "response.completed":')
    print('        break')
    print()
    print("  # Key difference vs Chat Completions: events are named and typed,")
    print("  # not opaque chunks. You dispatch on event.type, not finish_reason.")

    # Show actual mock events for context
    client = get_client()
    print()
    print("Mock stream events for reference:")
    with client.responses.create(model=MODEL, input="demo", stream=True) as stream:
        for evt in stream:
            print(f"  {evt.type}")


def anti_pattern_buffer_all_events() -> None:
    """ANTI-PATTERN 1: Buffering all events before processing defeats streaming.

    Collecting all events into a list before doing anything with them removes
    the latency benefit of streaming — the user sees nothing until the full
    response is ready, which is equivalent to non-streaming.
    """
    print("ANTI-PATTERN: Buffering all events before processing")
    print()

    client = get_client()

    # WRONG: collect everything first
    events: list[MockResponseEvent] = []
    with client.responses.create(model=MODEL, input="Hello", stream=True) as stream:
        for event in stream:
            events.append(event)  # buffering defeats streaming latency benefit

    # Now process — but user already waited for the full response
    for event in events:
        if event.type == "response.content_part.delta":
            print(f"  (too late) delta: {event.delta!r}")

    print()
    print("  Fix: process each event inside the loop as it arrives.")
    print("  RIGHT:")
    print("    with client.responses.create(..., stream=True) as stream:")
    print("        for event in stream:")
    print('            if event.type == "response.content_part.delta":')
    print("                print(event.delta, end='', flush=True)  # immediate output")


if __name__ == "__main__":
    sep = "=" * 60
    mode = "MOCK" if is_mock() else "LIVE"
    print(f"{NL}{sep}{NL}Domain 4 - Task 4.7: Responses API Streaming [{mode}]{NL}{sep}")

    print(f"{NL}--- DEMO 1: Basic Streaming Events ---")
    demo_basic_streaming_events()

    print(f"{NL}--- DEMO 2: Accumulating Text Deltas ---")
    demo_accumulate_text_deltas()

    print(f"{NL}--- DEMO 3: Tool Call Events Pattern ---")
    demo_tool_call_events()

    print(f"{NL}--- ANTI-PATTERN 1: Buffering All Events ---")
    anti_pattern_buffer_all_events()

    print(f"{NL}KEY TAKEAWAYS:")
    print("  1. client.responses.create(stream=True) returns an event iterator, not chunks")
    print("  2. Dispatch on event.type: 'response.content_part.delta' carries text increments")
    print("  3. 'response.completed' carries the final output_text for the whole response")
    print("  4. Responses API events are named/typed; Chat stream chunks are opaque objects")
    print("  5. Process events immediately in the loop — never buffer then process")
