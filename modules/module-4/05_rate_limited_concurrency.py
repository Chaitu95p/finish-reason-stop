"""Domain 4 - Task 4.5: Rate-Limited Concurrency with Semaphore

CONCEPTS:
  1. asyncio.Semaphore — limit concurrent coroutines
  2. async with sem: — acquire before API call
  3. Token bucket pattern — requests per minute limit
  4. Backpressure — queue fills, workers throttle

Mnemonic: SLAT — Semaphore, Limit, Async, Throttle

Run:
  uv run python 05_rate_limited_concurrency.py
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass

from shared.mock import get_async_client, is_mock

NL = chr(10)
MODEL = "gpt-4o"

# Simulated per-request delay so slot occupancy is visible
_REQUEST_DELAY_S = 0.1
MAX_CONCURRENCY = 3
TOTAL_REQUESTS = 10


@dataclass
class SlottedResult:
    """Result enriched with which semaphore slot was occupied."""

    request_id: int
    response: str
    slot_label: str
    elapsed_ms: float
    error: str | None = None


async def request_with_semaphore(
    sem: asyncio.Semaphore,
    request_id: int,
    prompt: str,
    active_slots: list[int],
) -> SlottedResult:
    """
    Acquire the semaphore before making an API call, release after.

    active_slots is a mutable list tracking currently occupied slot indices.
    We assign the lowest free slot number for display purposes.
    """
    async with sem:
        # Determine which slot we occupy (first gap in active_slots)
        occupied = set(active_slots)
        slot_num = next(i for i in range(MAX_CONCURRENCY) if i not in occupied)
        active_slots.append(slot_num)
        slot_label = f"slot-{slot_num}"

        print(f"  [{slot_label}] request-{request_id:02d} STARTED  (active: {sorted(active_slots)})")

        start = time.perf_counter()
        try:
            await asyncio.sleep(_REQUEST_DELAY_S)  # Simulate I/O time
            client = await get_async_client()
            response = await client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
            )
            content: str = response.choices[0].message.content or ""
            elapsed_ms = (time.perf_counter() - start) * 1000

            print(f"  [{slot_label}] request-{request_id:02d} DONE     {elapsed_ms:.1f}ms")
            return SlottedResult(
                request_id=request_id,
                response=content,
                slot_label=slot_label,
                elapsed_ms=elapsed_ms,
            )
        finally:
            active_slots.remove(slot_num)


async def run_rate_limited(prompts: list[str], max_concurrent: int) -> list[SlottedResult]:
    """Run all prompts with at most max_concurrent active at once."""
    sem = asyncio.Semaphore(max_concurrent)
    active_slots: list[int] = []

    tasks = [
        request_with_semaphore(sem, i, prompt, active_slots)
        for i, prompt in enumerate(prompts)
    ]
    results: list[SlottedResult] = await asyncio.gather(*tasks)
    return list(results)


async def run_unlimited(prompts: list[str]) -> tuple[list[str], float]:
    """Run all prompts concurrently without any limit (for comparison)."""
    client = await get_async_client()

    async def _one(prompt: str) -> str:
        await asyncio.sleep(_REQUEST_DELAY_S)
        resp = await client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content or ""

    start = time.perf_counter()
    responses: list[str] = await asyncio.gather(*[_one(p) for p in prompts])
    elapsed = (time.perf_counter() - start) * 1000
    return responses, elapsed


# ANTI-PATTERN 1: Acquiring semaphore outside the async with block
async def anti_pattern_manual_acquire(sem: asyncio.Semaphore, prompt: str) -> str:
    """
    Wrong: manually calling sem.acquire() without a matching release in a try/finally.

    If an exception occurs between acquire and release, the semaphore leaks —
    future tasks will hang forever waiting for a slot that never frees.
    """
    await sem.acquire()
    try:
        client = await get_async_client()
        # If this raises, sem.release() below may never run (if not in finally)
        resp = await client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content or ""
    except Exception:
        raise
    finally:
        sem.release()  # Correct in this version, but fragile — use 'async with' instead


async def main() -> None:
    """Entry point for semaphore-based rate limiting demos."""
    mock_label = "[MOCK]" if is_mock() else "[LIVE]"
    sep = "=" * 60

    print(f"Domain 4 - Task 4.5: Rate-Limited Concurrency with Semaphore {mock_label}")
    print(sep)

    prompts = [f"Summarize concept #{i}" for i in range(TOTAL_REQUESTS)]
    print(f"Total requests : {TOTAL_REQUESTS}")
    print(f"Max concurrent : {MAX_CONCURRENCY}")
    print(f"Request delay  : {_REQUEST_DELAY_S * 1000:.0f}ms each")

    print(sep)

    # DEMO 1: Rate-limited with Semaphore(3)
    print(f"DEMO 1: Semaphore({MAX_CONCURRENCY}) — at most {MAX_CONCURRENCY} active at once")
    print(sep)
    start = time.perf_counter()
    results = await run_rate_limited(prompts, max_concurrent=MAX_CONCURRENCY)
    rate_limited_ms = (time.perf_counter() - start) * 1000

    print(sep)
    print(f"Completed {len(results)} requests in {rate_limited_ms:.1f}ms")

    print(sep)

    # DEMO 2: Unlimited concurrency (all 10 at once) for comparison
    print(f"DEMO 2: Unlimited concurrency ({TOTAL_REQUESTS} requests simultaneously)")
    print(sep)
    _, unlimited_ms = await run_unlimited(prompts)
    print(f"Unlimited total: {unlimited_ms:.1f}ms")

    print(sep)
    batches = -(-TOTAL_REQUESTS // MAX_CONCURRENCY)  # ceiling division
    expected_limited = batches * _REQUEST_DELAY_S * 1000
    print("Comparison:")
    print(f"  Unlimited    : {unlimited_ms:.1f}ms  (~1 batch of {TOTAL_REQUESTS})")
    print(f"  Rate-limited : {rate_limited_ms:.1f}ms  (~{batches} batches of {MAX_CONCURRENCY})")
    print(f"  Expected limited time: ~{expected_limited:.0f}ms ({batches} x {_REQUEST_DELAY_S * 1000:.0f}ms)")
    print(
        f"{NL}Semaphore backpressure: once all {MAX_CONCURRENCY} slots are filled,"
        f" new tasks wait in the asyncio queue until a slot is released."
    )

    print(sep)

    # ANTI-PATTERN 1 — manual acquire/release
    print("ANTI-PATTERN 1: Manual sem.acquire() without 'async with'")
    print(sep)
    demo_sem = asyncio.Semaphore(1)
    result = await anti_pattern_manual_acquire(demo_sem, "test prompt")
    print(f"Result: {result!r}")
    print(
        f"Even though try/finally works here, it is verbose and error-prone.{NL}"
        f"Fix: use 'async with sem:' — it guarantees release even on exceptions."
    )

    print(sep)
    print("KEY TAKEAWAYS:")
    print("  - asyncio.Semaphore(N) limits concurrent coroutines to N at a time")
    print("  - 'async with sem:' is the safe pattern — always releases on exit")
    print("  - Rate-limiting prevents hitting API rate limits (e.g. 3 RPM or 60 RPM)")
    print("  - Backpressure: tasks queue naturally — no explicit queueing code needed")
    print("  - Tune N to stay under your API tier's requests-per-minute limit")


if __name__ == "__main__":
    asyncio.run(main())
