"""Exercise 4 - Task 4.2: Rate-Limited Batch Processing

GOAL: Process 20 items with bounded concurrency (asyncio.Semaphore)
to stay within rate limits while maximizing throughput.

SKILLS PRACTICED:
  - asyncio.Semaphore for concurrency control
  - Tracking per-item results and failures
  - Throughput vs latency tradeoff

Run:
  uv run python 02_rate_limited_batch.py
"""

NL = chr(10)
MODEL = "gpt-4o-mini"

import asyncio
import time
from dataclasses import dataclass

from shared.mock import get_async_client, is_mock

ITEMS = [f"Translate 'hello' to {lang}" for lang in [
    "Spanish", "French", "German", "Italian", "Portuguese",
    "Dutch", "Swedish", "Norwegian", "Danish", "Finnish",
    "Polish", "Czech", "Hungarian", "Romanian", "Bulgarian",
    "Greek", "Turkish", "Arabic", "Hebrew", "Japanese",
]]


@dataclass
class BatchResult:
    item: str
    result: str | None
    error: str | None


async def process_item(
    client: object,
    semaphore: asyncio.Semaphore,
    item: str,
) -> BatchResult:
    """Process a single item, respecting the concurrency limit."""
    async with semaphore:
        try:
            response = await client.chat.completions.create(  # type: ignore[attr-defined]
                model=MODEL,
                messages=[{"role": "user", "content": item}],
                max_tokens=20,
            )
            content = response.choices[0].message.content or ""
            return BatchResult(item=item, result=content, error=None)
        except Exception as e:
            return BatchResult(item=item, result=None, error=str(e))


async def process_batch(items: list[str], max_concurrent: int = 5) -> list[BatchResult]:
    """Process all items with bounded concurrency."""
    client = await get_async_client()
    semaphore = asyncio.Semaphore(max_concurrent)
    tasks = [process_item(client, semaphore, item) for item in items]
    return await asyncio.gather(*tasks)


if __name__ == "__main__":
    sep = "=" * 60
    mode = "MOCK" if is_mock() else "LIVE"
    print(f"{NL}{sep}{NL}Exercise 4 - Task 4.2: Rate-Limited Batch [{mode}]{NL}{sep}")

    t0 = time.monotonic()
    results = asyncio.run(process_batch(ITEMS, max_concurrent=5))
    elapsed = time.monotonic() - t0

    successful = [r for r in results if r.error is None]
    failed = [r for r in results if r.error is not None]

    print(f"{NL}  Processed {len(ITEMS)} items in {elapsed*1000:.1f}ms")
    print(f"  Successful: {len(successful)}")
    print(f"  Failed:     {len(failed)}")
    print(f"{NL}  First 5 results:")
    for r in results[:5]:
        print(f"    {r.item[:40]!r} → {r.result!r}")

    print(f"{NL}KEY TAKEAWAYS:")
    print("  1. asyncio.Semaphore(N) limits concurrent API calls to N at any moment")
    print("  2. asyncio.gather() still queues all tasks — semaphore controls concurrency")
    print("  3. Each item gets a BatchResult with result or error — no crashes on partial failure")
