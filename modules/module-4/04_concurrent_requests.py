"""Domain 4 - Task 4.4: Concurrent Requests with asyncio.gather()

CONCEPTS:
  1. asyncio.gather() — run multiple coroutines concurrently
  2. Parallel API calls — N calls in time of 1 (when not rate limited)
  3. Collecting results — gather returns list in input order
  4. Exception handling in gather — return_exceptions=True

Mnemonic: GPCE — Gather, Parallel, Collect, Exceptions

Run:
  uv run python 04_concurrent_requests.py
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass

from shared.mock import get_async_client, is_mock

NL = chr(10)
MODEL = "gpt-4o"

# Simulated per-request delay to make concurrency benefit visible with the mock
_REQUEST_DELAY_S = 0.15


@dataclass
class RequestResult:
    """Result of a single async completion request."""

    prompt: str
    response: str
    elapsed_ms: float
    error: str | None = None


async def make_request(prompt: str, request_index: int) -> RequestResult:
    """Make one async completion request, adding a small delay to show parallelism."""
    client = await get_async_client()
    start = time.perf_counter()

    # Simulated delay so sequential vs concurrent timing difference is visible
    await asyncio.sleep(_REQUEST_DELAY_S)

    response = await client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    elapsed_ms = (time.perf_counter() - start) * 1000
    content: str = response.choices[0].message.content or ""
    prompt_preview = repr(prompt)[:40]
    print(f"  [request {request_index}] done in {elapsed_ms:.1f}ms — {prompt_preview}")
    return RequestResult(prompt=prompt, response=content, elapsed_ms=elapsed_ms)


async def run_sequential(prompts: list[str]) -> tuple[list[RequestResult], float]:
    """Run requests one after another and return results + total elapsed time."""
    start = time.perf_counter()
    results: list[RequestResult] = []
    for i, prompt in enumerate(prompts):
        result = await make_request(prompt, request_index=i)
        results.append(result)
    total_ms = (time.perf_counter() - start) * 1000
    return results, total_ms


async def run_concurrent(prompts: list[str]) -> tuple[list[RequestResult], float]:
    """Run requests concurrently with asyncio.gather() and return results + total elapsed."""
    start = time.perf_counter()
    coroutines = [make_request(prompt, request_index=i) for i, prompt in enumerate(prompts)]
    results: list[RequestResult] = await asyncio.gather(*coroutines)
    total_ms = (time.perf_counter() - start) * 1000
    return list(results), total_ms


async def run_concurrent_with_exceptions(
    prompts: list[str],
) -> list[RequestResult | BaseException]:
    """
    Gather with return_exceptions=True — failed tasks return the exception instead of raising.

    Useful when some requests can fail without cancelling the entire batch.
    """
    coroutines = [make_request(prompt, request_index=i) for i, prompt in enumerate(prompts)]
    raw: list[RequestResult | BaseException] = await asyncio.gather(
        *coroutines, return_exceptions=True
    )
    return raw


async def demo_exception_handling() -> None:
    """Show return_exceptions=True collecting both successes and failures."""

    async def failing_request(label: str) -> str:
        await asyncio.sleep(0.05)
        raise ValueError(f"Simulated error for {label!r}")

    async def ok_request(label: str) -> str:
        await asyncio.sleep(0.05)
        return f"OK result for {label!r}"

    mixed: list[str | BaseException] = await asyncio.gather(
        ok_request("task-A"),
        failing_request("task-B"),
        ok_request("task-C"),
        return_exceptions=True,
    )
    for i, item in enumerate(mixed):
        if isinstance(item, BaseException):
            print(f"  Task {i}: FAILED — {type(item).__name__}: {item}")
        else:
            print(f"  Task {i}: SUCCESS — {item!r}")


async def main() -> None:
    """Entry point for all concurrent request demos."""
    mock_label = "[MOCK]" if is_mock() else "[LIVE]"
    sep = "=" * 60

    print(f"Domain 4 - Task 4.4: Concurrent Requests with asyncio.gather() {mock_label}")
    print(sep)

    prompts = [
        "What is Python?",
        "What is asyncio?",
        "What is a coroutine?",
        "What is an event loop?",
        "What is asyncio.gather()?",
    ]
    print(f"Running {len(prompts)} requests sequentially then concurrently.")
    print(f"(Each request has a {_REQUEST_DELAY_S * 1000:.0f}ms simulated delay.)")

    print(sep)

    # DEMO 1: Sequential
    print("DEMO 1: Sequential execution")
    print(sep)
    seq_results, seq_total = await run_sequential(prompts)
    print(f"Sequential total: {seq_total:.1f}ms")

    print(sep)

    # DEMO 2: Concurrent with gather
    print("DEMO 2: Concurrent execution with asyncio.gather()")
    print(sep)
    con_results, con_total = await run_concurrent(prompts)
    print(f"Concurrent total: {con_total:.1f}ms")

    print(sep)
    speedup = seq_total / con_total if con_total > 0 else 0
    print(f"Sequential  : {seq_total:.1f}ms")
    print(f"Concurrent  : {con_total:.1f}ms")
    print(f"Speedup     : {speedup:.1f}x (expected ~{len(prompts)}x)")
    print("Results are in INPUT ORDER regardless of which completed first:")
    for i, r in enumerate(con_results):
        p_prev = repr(r.prompt)[:40]
        r_prev = repr(r.response)[:30]
        print(f"  [{i}] {p_prev} => {r_prev}")

    print(sep)

    # DEMO 3: return_exceptions=True
    print("DEMO 3: asyncio.gather(return_exceptions=True) — mixed success/failure")
    print(sep)
    await demo_exception_handling()

    print(sep)
    print("KEY TAKEAWAYS:")
    print("  - asyncio.gather(*coroutines) runs all coroutines concurrently")
    print("  - Results are returned in INPUT order, not completion order")
    print("  - N concurrent I/O requests take ~1x time instead of ~Nx time")
    print("  - return_exceptions=True prevents one failure from cancelling all tasks")
    print("  - Concurrency is cooperative — the event loop switches at each 'await' point")


if __name__ == "__main__":
    asyncio.run(main())
