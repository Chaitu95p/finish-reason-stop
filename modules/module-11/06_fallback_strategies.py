"""Domain 11 - Task 11.6: Fallback Strategies

CONCEPTS:
  1. Model fallback chain — try gpt-4o, fallback to gpt-4o-mini
  2. Cached response fallback — return last known-good response on failure
  3. Degraded mode — return a structured error response, not a crash
  4. Timeout-based fallback — budget a short time, fall back if exceeded

Mnemonic: MCDT — Model_chain, Cache, Degraded, Timeout_budget

Run:
  uv run python 06_fallback_strategies.py
"""

NL = chr(10)

from typing import Any

import openai
from shared.mock import get_client, is_mock

MODEL_CHAIN = ["gpt-4o", "gpt-4o-mini"]

RESPONSE_CACHE: dict[str, str] = {}


def call_with_model_fallback(client: object, prompt: str) -> str:
    """Try models in chain order; return first successful response."""
    last_error: Exception | None = None
    for model in MODEL_CHAIN:
        try:
            resp = client.chat.completions.create(  # type: ignore[attr-defined]
                model=model,
                messages=[{"role": "user", "content": prompt}],
                timeout=15.0,
            )
            content = resp.choices[0].message.content or ""
            print(f"    Success with model: {model!r}")
            return content
        except (openai.RateLimitError, openai.InternalServerError) as e:
            print(f"    {model!r} failed ({type(e).__name__}), trying next...")
            last_error = e
    raise RuntimeError(f"All models in chain failed. Last: {last_error}") from last_error


def call_with_cache_fallback(client: object, prompt: str, cache_key: str) -> str:
    """Try API; return cached response if API fails."""
    try:
        resp = client.chat.completions.create(  # type: ignore[attr-defined]
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
        )
        content = resp.choices[0].message.content or ""
        RESPONSE_CACHE[cache_key] = content
        return content
    except openai.APIError as e:
        cached = RESPONSE_CACHE.get(cache_key)
        if cached:
            print(f"    API failed ({type(e).__name__}), returning cached response")
            return f"[CACHED] {cached}"
        raise


def call_with_degraded_mode(client: object, prompt: str) -> dict[str, Any]:
    """Return a structured result with success/error indication."""
    try:
        resp = client.chat.completions.create(  # type: ignore[attr-defined]
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
        )
        return {
            "success": True,
            "content": resp.choices[0].message.content or "",
            "model": "gpt-4o",
            "error": None,
        }
    except openai.RateLimitError:
        return {"success": False, "content": None, "model": None, "error": "rate_limited"}
    except openai.APIConnectionError:
        return {"success": False, "content": None, "model": None, "error": "connection_error"}
    except openai.APIError as e:
        return {"success": False, "content": None, "model": None, "error": str(type(e).__name__)}


def demo_model_fallback() -> None:
    """DEMO 1: Model fallback chain."""
    client = get_client()
    prompt = "What is the capital of France?"
    content = call_with_model_fallback(client, prompt)
    print(f"  Result: {content!r}")


def demo_cache_fallback() -> None:
    """DEMO 2: Cache fallback on API failure."""
    client = get_client()
    prompt = "What is 2 + 2?"
    cache_key = "basic_math"

    # First call succeeds and populates cache
    result1 = call_with_cache_fallback(client, prompt, cache_key)
    print(f"  Call 1 (fresh): {result1!r}")
    print(f"  Cache populated: {cache_key!r} → {RESPONSE_CACHE.get(cache_key)!r}")

    # Second call uses cache (simulated by clearing with cached value already in)
    result2 = call_with_cache_fallback(client, prompt, cache_key)
    print(f"  Call 2 (from API): {result2!r}")


def demo_degraded_mode() -> None:
    """DEMO 3: Degraded mode — structured result with error field."""
    client = get_client()
    result = call_with_degraded_mode(client, "Tell me a joke.")
    print(f"  success: {result['success']}")
    print(f"  content: {result['content']!r}")
    print(f"  error: {result['error']!r}")
    print("  Caller checks result['success'] before using result['content']")


def demo_strategy_matrix() -> None:
    """DEMO 4: Which fallback strategy to use when."""
    print("  Fallback strategy selection:")
    print("    Model chain:     Best when gpt-4o is rate-limited; gpt-4o-mini acceptable")
    print("    Cache fallback:  Best for read-heavy, idempotent, stable prompts")
    print("    Degraded mode:   Best when partial success is acceptable to callers")
    print("    Timeout budget:  Best for latency-sensitive paths (user-facing)")
    print(f"{NL}  Anti-pattern: hiding all errors with try/except and returning empty string")
    print("    → caller can't distinguish 'no content' from 'API failed'")


if __name__ == "__main__":
    sep = "=" * 60
    mode = "MOCK" if is_mock() else "LIVE"
    print(f"{NL}{sep}{NL}Domain 11 - Task 11.6: Fallback Strategies [{mode}]{NL}{sep}")

    print(f"{NL}--- DEMO 1: Model Fallback Chain ---")
    demo_model_fallback()

    print(f"{NL}--- DEMO 2: Cache Fallback ---")
    demo_cache_fallback()

    print(f"{NL}--- DEMO 3: Degraded Mode ---")
    demo_degraded_mode()

    print(f"{NL}--- DEMO 4: Strategy Matrix ---")
    demo_strategy_matrix()

    print(f"{NL}KEY TAKEAWAYS:")
    print("  1. Model chain: try premium model first, fall back to cheaper on rate limit")
    print("  2. Cache fallback: always update cache on success; serve stale on failure")
    print("  3. Degraded mode: return {'success': bool, 'error': str} instead of raising")
    print("  4. Never swallow errors with bare except returning empty string")
