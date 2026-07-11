"""Exercise 4 - Task 4.1: Parallel Subagents with asyncio.gather

GOAL: Run N independent analysis tasks concurrently using AsyncClient
and asyncio.gather(). Compare wall-clock time vs sequential.

SKILLS PRACTICED:
  - AsyncOpenAI / AsyncMockClient
  - asyncio.gather() for true parallelism
  - Wall-clock timing of concurrent vs sequential

Run:
  uv run python 01_parallel_subagents.py
"""

NL = chr(10)
MODEL = "gpt-4o-mini"

import asyncio
import time

from shared.mock import get_async_client, is_mock

ANALYSIS_TASKS = [
    ("France", "Name the capital and one famous landmark."),
    ("Japan", "Name the capital and one famous landmark."),
    ("Brazil", "Name the capital and one famous landmark."),
    ("Egypt", "Name the capital and one famous landmark."),
    ("Canada", "Name the capital and one famous landmark."),
]


async def analyze(client: object, country: str, prompt: str) -> tuple[str, str]:
    """Single async analysis task."""
    response = await client.chat.completions.create(  # type: ignore[attr-defined]
        model=MODEL,
        messages=[{"role": "user", "content": f"{country}: {prompt}"}],
        max_tokens=50,
    )
    return country, response.choices[0].message.content or ""


async def run_parallel(tasks: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Run all analysis tasks concurrently."""
    client = await get_async_client()
    coroutines = [analyze(client, country, prompt) for country, prompt in tasks]
    return await asyncio.gather(*coroutines)


async def run_sequential(tasks: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Run all analysis tasks one by one."""
    client = await get_async_client()
    results = []
    for country, prompt in tasks:
        result = await analyze(client, country, prompt)
        results.append(result)
    return results


if __name__ == "__main__":
    sep = "=" * 60
    mode = "MOCK" if is_mock() else "LIVE"
    print(f"{NL}{sep}{NL}Exercise 4 - Task 4.1: Parallel Subagents [{mode}]{NL}{sep}")

    t0 = time.monotonic()
    parallel_results = asyncio.run(run_parallel(ANALYSIS_TASKS))
    parallel_time = time.monotonic() - t0

    t0 = time.monotonic()
    sequential_results = asyncio.run(run_sequential(ANALYSIS_TASKS))
    sequential_time = time.monotonic() - t0

    print(f"{NL}  Parallel results ({parallel_time*1000:.1f}ms):")
    for country, answer in parallel_results:
        print(f"    {country}: {answer!r}")

    print(f"{NL}  Sequential time: {sequential_time*1000:.1f}ms")
    speedup = sequential_time / parallel_time if parallel_time > 0 else float("inf")
    print(f"  Speedup: {speedup:.1f}×")

    print(f"{NL}KEY TAKEAWAYS:")
    print("  1. asyncio.gather() runs coroutines concurrently in one event loop")
    print("  2. Wall-clock time ≈ slowest single task (not sum of all tasks)")
    print("  3. Use AsyncMockClient / AsyncOpenAI consistently — don't mix sync/async")
