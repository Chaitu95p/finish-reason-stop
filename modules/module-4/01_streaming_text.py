"""Domain 4 - Task 4.1: Streaming Text Responses

CONCEPTS:
  1. stream=True in chat.completions.create() — returns iterator
  2. Iterating chunks — chunk.choices[0].delta.content
  3. None content in final chunk — end of stream marker
  4. Latency tradeoff — first token faster, total time similar

Mnemonic: SCDF — Stream, Chunks, Delta, Final_chunk

Run:
  uv run python 01_streaming_text.py
"""

from __future__ import annotations

import time

from shared.mock import get_client, is_mock

NL = chr(10)
MODEL = "gpt-4o"


def stream_and_accumulate(prompt: str) -> str:
    """Stream a completion for prompt, print each chunk, return full text."""
    client = get_client()
    accumulated: list[str] = []
    chunk_count = 0

    print(f"Streaming response for: {prompt!r}")
    print("-" * 40)

    with client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    ) as stream:
        for chunk in stream:
            choice = chunk.choices[0]
            delta_content: str | None = choice.delta.content
            finish_reason: str | None = choice.finish_reason

            if delta_content is not None:
                # Print each chunk as it arrives — simulates real-time streaming UX
                print(delta_content, end="", flush=True)
                accumulated.append(delta_content)
                chunk_count += 1
            elif finish_reason is not None:
                # Final chunk signals end of stream — content is None here
                print(f"{NL}[stream ended, finish_reason={finish_reason!r}]")

    full_text = "".join(accumulated)
    print(f"Total chunks received: {chunk_count}")
    print(f"Full response: {full_text!r}")
    return full_text


def time_stream_to_first_token(prompt: str) -> float:
    """Measure time from request start to first token received."""
    client = get_client()
    start = time.perf_counter()
    first_token_time: float | None = None

    with client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    ) as stream:
        for chunk in stream:
            delta_content: str | None = chunk.choices[0].delta.content
            if delta_content is not None and first_token_time is None:
                first_token_time = time.perf_counter() - start
                break

    elapsed = first_token_time or (time.perf_counter() - start)
    return elapsed


# ANTI-PATTERN 1: Collecting all chunks before processing defeats streaming purpose
def anti_pattern_collect_then_process(prompt: str) -> str:
    """Wrong: wait for all chunks before doing anything — kills latency benefit."""
    client = get_client()
    all_chunks = []

    with client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    ) as stream:
        for chunk in stream:
            all_chunks.append(chunk)  # Wait for everything first

    # Only now process — user saw nothing until fully done
    text = "".join(
        c.choices[0].delta.content or ""
        for c in all_chunks
        if c.choices[0].delta.content is not None
    )
    return text


if __name__ == "__main__":
    mock_label = "[MOCK]" if is_mock() else "[LIVE]"
    sep = "=" * 60

    print(f"Domain 4 - Task 4.1: Streaming Text Responses {mock_label}")
    print(sep)

    # DEMO 1: Basic streaming with chunk-by-chunk output
    print("DEMO 1: Streaming text, printing each chunk as it arrives")
    print(sep)
    result = stream_and_accumulate("What is the capital of France?")

    print(sep)

    # DEMO 2: Measuring time to first token
    print("DEMO 2: Time to first token measurement")
    print(sep)
    ttft = time_stream_to_first_token("Tell me a joke.")
    print(f"Time to first token: {ttft * 1000:.2f}ms")
    print(
        "Streaming benefit: UI can start rendering immediately, "
        "rather than waiting for full response."
    )

    print(sep)

    # ANTI-PATTERN 1
    print("ANTI-PATTERN 1: Collecting all chunks before processing")
    print(sep)
    print("This approach defeats the purpose of streaming:")
    ap_result = anti_pattern_collect_then_process("Why is the sky blue?")
    print(f"Result (only shown after all chunks collected): {ap_result!r}")
    print(
        "Problem: user sees blank screen until the ENTIRE response is received."
        f"{NL}Fix: process and display each chunk as soon as it arrives."
    )

    print(sep)
    print("KEY TAKEAWAYS:")
    print("  - stream=True returns a context manager; iterate chunks inside 'with' block")
    print("  - chunk.choices[0].delta.content is None on the final chunk (finish_reason set)")
    print("  - Streaming lowers perceived latency: first token appears almost immediately")
    print("  - Total time is similar to non-streaming; the UX benefit is in responsiveness")
    print("  - Always process chunks as they arrive — never buffer all before rendering")
