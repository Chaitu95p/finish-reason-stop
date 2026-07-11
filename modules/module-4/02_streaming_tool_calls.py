"""Domain 4 - Task 4.2: Streaming with Tool Calls

CONCEPTS:
  1. Tool call deltas — chunk.choices[0].delta.tool_calls
  2. Accumulating function arguments — delta.function.arguments fragments
  3. Detecting tool call completion — finish_reason in final chunk
  4. Parsing accumulated JSON after stream ends

Mnemonic: TACD — Tool_calls, Accumulate, Complete, Decode

Run:
  uv run python 02_streaming_tool_calls.py
"""

from __future__ import annotations

import json
import types
from dataclasses import dataclass, field

from shared.mock import get_client, is_mock

NL = chr(10)
MODEL = "gpt-4o"


# ---------------------------------------------------------------------------
# Domain objects
# ---------------------------------------------------------------------------


@dataclass
class AccumulatedToolCall:
    """Holds incrementally assembled tool call data from stream deltas."""

    index: int
    id: str = ""
    name: str = ""
    arguments_fragments: list[str] = field(default_factory=list)

    @property
    def arguments_raw(self) -> str:
        return "".join(self.arguments_fragments)

    def parse_arguments(self) -> dict:  # type: ignore[type-arg]
        """Parse the accumulated argument string as JSON."""
        raw = self.arguments_raw
        if not raw:
            return {}
        return json.loads(raw)  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# Helpers to build synthetic tool-call delta chunks
# (The mock client stream yields text deltas, not tool_call deltas.
#  We hand-craft the sequence to demonstrate the accumulation pattern.)
# ---------------------------------------------------------------------------


def _make_tool_chunk(
    index: int,
    call_id: str | None,
    fn_name: str | None,
    args_fragment: str | None,
    finish_reason: str | None = None,
) -> object:
    """Build a duck-typed stream chunk carrying a tool_call delta."""
    fn_delta = types.SimpleNamespace(
        name=fn_name,
        arguments=args_fragment,
    )
    tool_call_delta = types.SimpleNamespace(
        index=index,
        id=call_id,
        type="function" if call_id else None,
        function=fn_delta,
    )
    delta = types.SimpleNamespace(
        content=None,
        tool_calls=[tool_call_delta],
    )
    choice = types.SimpleNamespace(
        delta=delta,
        finish_reason=finish_reason,
        index=0,
    )
    return types.SimpleNamespace(
        choices=[choice],
        id="chatcmpl-mock-tool-stream",
        model=MODEL,
    )


def _make_final_chunk(finish_reason: str = "tool_calls") -> object:
    """Final chunk — no delta content, just finish_reason."""
    delta = types.SimpleNamespace(content=None, tool_calls=None)
    choice = types.SimpleNamespace(delta=delta, finish_reason=finish_reason, index=0)
    return types.SimpleNamespace(choices=[choice], id="chatcmpl-mock-final", model=MODEL)


def build_synthetic_tool_stream() -> list[object]:
    """
    Return a sequence of chunks that simulates a real tool-call stream.

    In a real stream the model emits:
      chunk 1 — tool_calls[0].id + function.name (role=assistant, index=0)
      chunk 2-N — tool_calls[0].function.arguments fragments
      final chunk — finish_reason="tool_calls", empty delta
    """
    return [
        # Chunk 1: call id + function name
        _make_tool_chunk(
            index=0,
            call_id="call_abc123",
            fn_name="get_weather",
            args_fragment=None,
        ),
        # Chunks 2-5: argument JSON fragmented across multiple deltas
        _make_tool_chunk(index=0, call_id=None, fn_name=None, args_fragment='{"loc'),
        _make_tool_chunk(index=0, call_id=None, fn_name=None, args_fragment='ation":'),
        _make_tool_chunk(index=0, call_id=None, fn_name=None, args_fragment=' "Paris"'),
        _make_tool_chunk(index=0, call_id=None, fn_name=None, args_fragment="}"),
        # Final chunk
        _make_final_chunk(finish_reason="tool_calls"),
    ]


# ---------------------------------------------------------------------------
# Core demo: accumulate tool call deltas
# ---------------------------------------------------------------------------


def accumulate_tool_call_stream(chunks: list[object]) -> list[AccumulatedToolCall]:
    """
    Process a stream of chunks with tool_call deltas.

    Returns the list of fully accumulated AccumulatedToolCall objects.
    This is the pattern used with the real OpenAI streaming API.
    """
    tool_calls: dict[int, AccumulatedToolCall] = {}
    final_finish_reason: str | None = None

    for chunk in chunks:
        choice = chunk.choices[0]  # type: ignore[attr-defined]
        finish_reason: str | None = choice.finish_reason

        if finish_reason is not None:
            final_finish_reason = finish_reason
            print(f"  [stream complete] finish_reason={finish_reason!r}")
            continue

        delta = choice.delta
        if not delta.tool_calls:
            continue

        for tc_delta in delta.tool_calls:
            idx: int = tc_delta.index

            if idx not in tool_calls:
                tool_calls[idx] = AccumulatedToolCall(index=idx)

            tc = tool_calls[idx]

            if tc_delta.id:
                tc.id = tc_delta.id
                print(f"  [chunk] new tool call id={tc_delta.id!r}")

            fn = tc_delta.function
            if fn.name:
                tc.name = fn.name
                print(f"  [chunk] function name={fn.name!r}")

            if fn.arguments:
                tc.arguments_fragments.append(fn.arguments)
                print(f"  [chunk] args fragment={fn.arguments!r}")

    print(f"Stream ended with finish_reason={final_finish_reason!r}")
    return list(tool_calls.values())


# ---------------------------------------------------------------------------
# ANTI-PATTERN: parsing JSON before stream is done
# ---------------------------------------------------------------------------


def anti_pattern_premature_json_parse(chunks: list[object]) -> dict | None:  # type: ignore[type-arg]
    """
    Wrong: try to json.loads() each argument fragment as it arrives.

    Fragments are incomplete JSON strings — this will always raise JSONDecodeError
    until the very last fragment arrives.
    """
    for chunk in chunks:
        delta = chunk.choices[0].delta  # type: ignore[attr-defined]
        if not delta.tool_calls:
            continue
        for tc_delta in delta.tool_calls:
            if tc_delta.function.arguments:
                fragment = tc_delta.function.arguments
                try:
                    result = json.loads(fragment)
                    return result  # type: ignore[no-any-return]
                except json.JSONDecodeError:
                    # This fires for EVERY partial fragment — wasteful and always fails
                    print(f"  [premature parse failed] fragment={fragment!r} is not valid JSON yet")
    return None


if __name__ == "__main__":
    mock_label = "[MOCK]" if is_mock() else "[LIVE]"
    sep = "=" * 60

    print(f"Domain 4 - Task 4.2: Streaming with Tool Calls {mock_label}")
    print(sep)

    # DEMO 1: Show the synthetic chunk sequence
    print("DEMO 1: Synthetic tool-call stream chunks")
    print(sep)
    synthetic_chunks = build_synthetic_tool_stream()
    print(f"Total chunks in simulated stream: {len(synthetic_chunks)}")
    for i, c in enumerate(synthetic_chunks):
        choice = c.choices[0]  # type: ignore[attr-defined]
        delta = choice.delta
        tc_info = ""
        if delta.tool_calls:
            tc = delta.tool_calls[0]
            tc_info = (
                f"id={tc.id!r} name={tc.function.name!r} args={tc.function.arguments!r}"
            )
        finish = f" finish_reason={choice.finish_reason!r}" if choice.finish_reason else ""
        print(f"  Chunk {i}: tool_calls={bool(delta.tool_calls)} {tc_info}{finish}")

    print(sep)

    # DEMO 2: Accumulate and parse the tool call
    print("DEMO 2: Accumulating tool call deltas across chunks")
    print(sep)
    accumulated = accumulate_tool_call_stream(synthetic_chunks)

    print()
    for tc in accumulated:
        parsed_args = tc.parse_arguments()
        print("Reconstructed tool call:")
        print(f"  id        = {tc.id!r}")
        print(f"  name      = {tc.name!r}")
        print(f"  raw args  = {tc.arguments_raw!r}")
        print(f"  parsed    = {parsed_args}")

    print(sep)

    # DEMO 3: Using get_client() with tools (non-streaming path for comparison)
    print("DEMO 3: Non-streaming tool call (for comparison)")
    print(sep)
    client = get_client()
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "City name"},
                    },
                    "required": ["location"],
                },
            },
        }
    ]
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "What is the weather in Paris?"}],
        tools=tools,
        tool_choice="required",
    )
    choice = response.choices[0]
    print(f"finish_reason: {choice.finish_reason}")
    if choice.message.tool_calls:
        tc = choice.message.tool_calls[0]
        print(f"Tool call id: {tc.id}")
        print(f"Function: {tc.function.name}")
        print(f"Arguments: {tc.function.arguments}")
        print("Non-streaming: arguments arrive complete, no accumulation needed.")

    print(sep)

    # ANTI-PATTERN 1
    print("ANTI-PATTERN 1: Parsing argument fragments before stream is complete")
    print(sep)
    anti_pattern_premature_json_parse(synthetic_chunks)
    print(
        f"Problem: JSON argument string is split across multiple chunks.{NL}"
        f"Each fragment is invalid JSON until all fragments are joined.{NL}"
        f"Fix: accumulate ALL argument fragments, then parse once after finish_reason is set."
    )

    print(sep)
    print("KEY TAKEAWAYS:")
    print("  - Tool call arguments arrive in fragments across multiple stream chunks")
    print("  - Accumulate ALL argument fragments before calling json.loads()")
    print("  - finish_reason='tool_calls' signals the stream is done and args are complete")
    print("  - Index tool calls by chunk.choices[0].delta.tool_calls[i].index")
    print("  - The mock client does not emit tool_call deltas natively — use synthetic chunks for testing")
