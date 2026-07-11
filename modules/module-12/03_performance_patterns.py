"""Domain 12 - Task 12.3: Performance Patterns

CONCEPTS:
  1. Client reuse — one OpenAI() instance per process, not per request
  2. Connection pooling — httpx reuses TCP connections automatically
  3. Async concurrency — asyncio.gather() for parallel independent calls
  4. max_tokens — bound response length for latency-sensitive paths

Mnemonic: RCAM — Reuse_client, Connections, Async, Max_tokens

Run:
  uv run python 03_performance_patterns.py
"""

NL = chr(10)
MODEL = "gpt-4o-mini"

import asyncio
import time

from shared.mock import get_client, is_mock


def demo_client_reuse() -> None:
    """DEMO 1: Reuse one client instance across all requests."""
    print("  CORRECT: one client instance per process")
    print("""
    # Create once at module level or in a DI container
    client = openai.OpenAI()

    def handle_request(prompt: str) -> str:
        response = client.chat.completions.create(...)
        return response.choices[0].message.content or \"\"
    """)
    print("  ANTI-PATTERN: creating a new client per request")
    print("""
    def handle_request(prompt: str) -> str:
        client = openai.OpenAI()  # ← new HTTP connection every time
        response = client.chat.completions.create(...)
    """)
    print("  Each new OpenAI() creates a new httpx.Client with a new connection pool.")
    print("  Reusing one client amortizes TLS handshake overhead.")


def demo_max_tokens_for_latency() -> None:
    """DEMO 2: max_tokens bounds response time for interactive paths."""
    client = get_client()
    t0 = time.monotonic()
    response = client.chat.completions.create(  # type: ignore[attr-defined]
        model=MODEL,
        messages=[{"role": "user", "content": "Name one famous inventor."}],
        max_tokens=20,  # bound response length → bound latency
    )
    latency_ms = (time.monotonic() - t0) * 1000
    content = response.choices[0].message.content or ""
    print(f"  Response ({latency_ms:.1f}ms): {content!r}")
    print("  max_tokens=20 prevents runaway responses on simple queries")


async def make_call_async(client: object, prompt: str) -> str:
    """Single async API call."""
    from shared.mock import get_async_client
    aclient = await get_async_client()
    resp = await aclient.chat.completions.create(  # type: ignore[attr-defined]
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content or ""


def demo_async_concurrency() -> None:
    """DEMO 3: asyncio.gather() for parallel requests."""
    prompts = ["Capital of France?", "Capital of Japan?", "Capital of Germany?"]

    async def run_parallel() -> list[str]:
        tasks = [make_call_async(None, p) for p in prompts]
        return await asyncio.gather(*tasks)

    t0 = time.monotonic()
    results = asyncio.run(run_parallel())
    elapsed = (time.monotonic() - t0) * 1000
    print(f"  {len(prompts)} parallel calls in {elapsed:.1f}ms (vs sequential ~3×)")
    for prompt, result in zip(prompts, results):
        print(f"    {prompt!r} → {result!r}")


def demo_connection_pooling() -> None:
    """DEMO 4: Connection pooling and keep-alive."""
    print("  httpx connection pooling (automatic with one client instance):")
    print("    • First request: TCP connect + TLS handshake (~50-100ms overhead)")
    print("    • Subsequent requests: reuse existing connection (~0ms overhead)")
    print("    • pool size: httpx default is 100 connections")
    print(f"{NL}  Tuning httpx limits:")
    print("""
    import httpx, openai

    client = openai.OpenAI(
        http_client=httpx.Client(
            limits=httpx.Limits(
                max_connections=200,
                max_keepalive_connections=50,
                keepalive_expiry=30,
            )
        )
    )
    """)


if __name__ == "__main__":
    sep = "=" * 60
    mode = "MOCK" if is_mock() else "LIVE"
    print(f"{NL}{sep}{NL}Domain 12 - Task 12.3: Performance Patterns [{mode}]{NL}{sep}")

    print(f"{NL}--- DEMO 1: Client Reuse ---")
    demo_client_reuse()

    print(f"{NL}--- DEMO 2: max_tokens for Latency ---")
    demo_max_tokens_for_latency()

    print(f"{NL}--- DEMO 3: Async Concurrency ---")
    demo_async_concurrency()

    print(f"{NL}--- DEMO 4: Connection Pooling ---")
    demo_connection_pooling()

    print(f"{NL}KEY TAKEAWAYS:")
    print("  1. One OpenAI() client per process — not per request")
    print("  2. set max_tokens on all latency-sensitive calls")
    print("  3. Use asyncio.gather() for independent parallel requests")
    print("  4. httpx reuses connections automatically when client is reused")
