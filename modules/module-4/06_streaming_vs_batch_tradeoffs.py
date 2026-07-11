"""Domain 4 - Task 4.6: Streaming vs Batch Tradeoffs

CONCEPTS:
  1. Streaming — lower latency to first token, good for UX
  2. Batch (non-streaming) — simpler parsing, atomic response
  3. Use cases — streaming for chat UI, batch for processing pipelines
  4. Cost — identical; latency differs, not price

Mnemonic: LABS — Latency, Atomic, Batch, Streaming

Run:
  uv run python 06_streaming_vs_batch_tradeoffs.py
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from shared.mock import get_client, is_mock

NL = chr(10)
MODEL = "gpt-4o"


# ---------------------------------------------------------------------------
# Domain objects
# ---------------------------------------------------------------------------


@dataclass
class TimingResult:
    """Captures timing metrics for a single completion approach."""

    approach: str
    time_to_first_token_ms: float
    time_to_complete_ms: float
    full_text: str
    chunk_count: int = 0


# ---------------------------------------------------------------------------
# Streaming approach
# ---------------------------------------------------------------------------


def run_streaming(prompt: str) -> TimingResult:
    """Stream a completion and record time-to-first-token and total time."""
    client = get_client()
    start = time.perf_counter()
    first_token_ms: float | None = None
    accumulated: list[str] = []
    chunk_count = 0

    with client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    ) as stream:
        for chunk in stream:
            delta_content: str | None = chunk.choices[0].delta.content
            if delta_content is not None:
                if first_token_ms is None:
                    first_token_ms = (time.perf_counter() - start) * 1000
                accumulated.append(delta_content)
                chunk_count += 1

    total_ms = (time.perf_counter() - start) * 1000
    return TimingResult(
        approach="streaming",
        time_to_first_token_ms=first_token_ms or total_ms,
        time_to_complete_ms=total_ms,
        full_text="".join(accumulated),
        chunk_count=chunk_count,
    )


# ---------------------------------------------------------------------------
# Batch (non-streaming) approach
# ---------------------------------------------------------------------------


def run_batch(prompt: str) -> TimingResult:
    """Non-streaming completion — receives full response atomically."""
    client = get_client()
    start = time.perf_counter()

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        stream=False,
    )

    total_ms = (time.perf_counter() - start) * 1000
    content: str = response.choices[0].message.content or ""

    return TimingResult(
        approach="batch",
        # In batch mode, first token = full response (atomic delivery)
        time_to_first_token_ms=total_ms,
        time_to_complete_ms=total_ms,
        full_text=content,
        chunk_count=1,
    )


# ---------------------------------------------------------------------------
# Decision matrix
# ---------------------------------------------------------------------------

DECISION_MATRIX = [
    ("Chat UI / interactive",       "streaming", "User sees tokens as generated — low perceived latency"),
    ("Pipeline / data extraction",  "batch",     "Atomic response is easier to parse, no assembly needed"),
    ("Structured output (JSON)",    "batch",     "Parse complete JSON once; streaming fragments are invalid"),
    ("Long document generation",    "streaming", "Show progress indicator; user can cancel early"),
    ("Automated testing / CI",      "batch",     "Simpler assertions; no streaming infrastructure needed"),
    ("Cost optimization",           "either",    "Identical cost; streaming does NOT save money"),
    ("Rate-limited pipelines",      "batch",     "Easier to batch multiple independent requests"),
    ("Real-time audio/voice",       "streaming", "Low latency critical; TTS needs tokens ASAP"),
]


def print_decision_matrix() -> None:
    """Print a formatted table of when to use streaming vs batch."""
    col1 = max(len(r[0]) for r in DECISION_MATRIX)
    col2 = max(len(r[1]) for r in DECISION_MATRIX)
    header = f"{'Use Case':<{col1}}  {'Approach':<{col2}}  Rationale"
    print(header)
    print("-" * len(header))
    for use_case, approach, rationale in DECISION_MATRIX:
        print(f"{use_case:<{col1}}  {approach:<{col2}}  {rationale}")


# ---------------------------------------------------------------------------
# ANTI-PATTERN
# ---------------------------------------------------------------------------


def anti_pattern_streaming_for_json_tool_call(prompt: str) -> None:
    """
    Wrong: using streaming when you need to parse JSON/tool-call output immediately.

    The model's JSON arrives fragmented. You MUST wait for the complete stream
    before parsing, which eliminates streaming's first-token latency advantage
    while adding complexity. Use batch mode instead.
    """
    import json

    client = get_client()
    accumulated_args = ""

    with client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    ) as stream:
        for chunk in stream:
            delta_content: str | None = chunk.choices[0].delta.content
            if delta_content is not None:
                accumulated_args += delta_content
                # Wrong: try to parse each fragment — always fails until the last one
                try:
                    _ = json.loads(accumulated_args)
                except json.JSONDecodeError:
                    pass  # Expected — fragment is not valid JSON yet

    # Finally have complete text — same as batch, but with more code
    try:
        parsed = json.loads(accumulated_args)
        print(f"  Parsed after full stream: {parsed}")
    except json.JSONDecodeError:
        print(f"  Could not parse (mock returns prose, not JSON): {accumulated_args!r}")

    print(
        f"  Problem: streamed and reassembled just to parse JSON — no latency benefit.{NL}"
        f"  Fix: use batch mode (stream=False) for structured output."
    )


if __name__ == "__main__":
    mock_label = "[MOCK]" if is_mock() else "[LIVE]"
    sep = "=" * 60

    print(f"Domain 4 - Task 4.6: Streaming vs Batch Tradeoffs {mock_label}")
    print(sep)

    prompt = "Explain the difference between sync and async programming in two sentences."

    # DEMO 1: Side-by-side timing
    print("DEMO 1: Side-by-side timing comparison")
    print(sep)
    streaming_result = run_streaming(prompt)
    batch_result = run_batch(prompt)

    print(f"{'Metric':<30} {'Streaming':>12} {'Batch':>12}")
    print("-" * 56)
    print(
        f"{'Time to first token (ms)':<30} "
        f"{streaming_result.time_to_first_token_ms:>12.2f} "
        f"{batch_result.time_to_first_token_ms:>12.2f}"
    )
    print(
        f"{'Total time (ms)':<30} "
        f"{streaming_result.time_to_complete_ms:>12.2f} "
        f"{batch_result.time_to_complete_ms:>12.2f}"
    )
    print(
        f"{'Chunks received':<30} "
        f"{streaming_result.chunk_count:>12} "
        f"{batch_result.chunk_count:>12}"
    )
    print(f"{NL}Streaming first token is faster; total time is similar.")
    print("Cost is IDENTICAL for both approaches.")

    print(sep)

    # DEMO 2: Streaming text output
    print("DEMO 2: Streaming output (tokens arrive progressively)")
    print(sep)
    client = get_client()
    print("Streamed: ", end="", flush=True)
    with client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    ) as stream:
        for chunk in stream:
            content: str | None = chunk.choices[0].delta.content
            if content is not None:
                print(content, end="", flush=True)
    print()

    print(sep)

    # DEMO 3: Batch output
    print("DEMO 3: Batch output (full response arrives at once)")
    print(sep)
    batch_response = get_client().chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        stream=False,
    )
    print(f"Batch: {batch_response.choices[0].message.content}")

    print(sep)

    # DEMO 4: Decision matrix
    print("DEMO 4: Decision matrix — when to use each approach")
    print(sep)
    print_decision_matrix()

    print(sep)

    # ANTI-PATTERN 1
    print("ANTI-PATTERN 1: Streaming when you need immediate JSON parsing")
    print(sep)
    anti_pattern_streaming_for_json_tool_call("Return a JSON object with key 'answer'.")

    print(sep)
    print("KEY TAKEAWAYS:")
    print("  - Streaming lowers time-to-first-token; total wall time is roughly the same")
    print("  - Cost is IDENTICAL — streaming does not change token pricing")
    print("  - Use streaming for interactive UX; use batch for pipelines and JSON parsing")
    print("  - Batch mode gives you an atomic, fully-formed response — simpler downstream logic")
    print("  - For structured outputs (tool calls, JSON mode), batch is almost always better")
