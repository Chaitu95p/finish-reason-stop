"""Domain 12 - Task 12.9: Prompt Caching

CONCEPTS:
  1. Automatic caching — OpenAI caches prompt prefixes >= 1024 tokens
  2. cached_tokens — usage.prompt_tokens_details.cached_tokens
  3. Cache-friendly prompt structure — stable content first, variable last
  4. Cost impact — cached tokens billed at 50% of normal input price
  5. Cache TTL — typically 5–10 minutes; warm vs cold cache behaviour

Mnemonic: ACCCT — Automatic, Cached_tokens, Cache_structure, Cost, TTL

Run:
  uv run python 09_prompt_caching.py
"""

from __future__ import annotations

from shared.mock import get_client, is_mock
from shared.tokens import estimate_cost, format_cost

NL = chr(10)
MODEL = "gpt-4o"

# A large stable system prompt (simulates a long document or context > 1024 tokens)
STABLE_SYSTEM_PROMPT = (
    "You are an expert software architect with 20 years of experience. "
    "You follow SOLID principles, advocate for clean code, and always consider "
    "maintainability, testability, and performance in your answers. "
    "When reviewing code you point out: (1) design pattern opportunities, "
    "(2) potential bugs, (3) performance bottlenecks, (4) security vulnerabilities. "
    "You cite the relevant section of 'Clean Code' or 'Designing Data-Intensive "
    "Applications' when applicable. You prefer concrete examples over abstract advice. "
    "Keep answers under 300 words unless the question requires more detail. "
    # Repeat to simulate a long stable context that would exceed 1024 tokens in production
    "Architecture notes: microservices, event-driven, CQRS, hexagonal architecture. "
    "Preferred languages: Python, Go, TypeScript. Preferred databases: PostgreSQL, Redis. "
)


def demo_reading_cached_tokens() -> None:
    """DEMO 1: Reading cached_tokens from usage after a repeated prompt."""
    client = get_client()
    messages = [
        {"role": "system", "content": STABLE_SYSTEM_PROMPT},
        {"role": "user", "content": "What is the difference between CQRS and Event Sourcing?"},
    ]

    # First call — cold cache
    resp1 = client.chat.completions.create(model=MODEL, messages=messages)
    cached1 = resp1.usage.prompt_tokens_details.cached_tokens
    print(f"First call  — cached_tokens: {cached1}  (cold cache)")

    # Second call with same prefix — warm cache
    resp2 = client.chat.completions.create(model=MODEL, messages=messages)
    cached2 = resp2.usage.prompt_tokens_details.cached_tokens
    print(f"Second call — cached_tokens: {cached2}  (warm cache)")
    print()
    print(f"  prompt_tokens        : {resp2.usage.prompt_tokens}")
    print(f"  of which cached      : {cached2}  (billed at 50% price)")
    print(f"  non-cached           : {resp2.usage.prompt_tokens - cached2}")


def demo_cache_friendly_layout() -> None:
    """DEMO 2: Cache-optimized prompt layout — stable prefix, variable suffix."""
    print("Cache-FRIENDLY layout (stable prefix first):")
    print("  messages = [")
    print('    {"role": "system",  "content": LARGE_STABLE_SYSTEM_PROMPT},  # 1024+ tokens')
    print('    {"role": "user",    "content": STATIC_CONTEXT_DOCS},          # large stable docs')
    print('    {"role": "user",    "content": user_question},                 # variable — last!')
    print("  ]")
    print()
    print("Cache-UNFRIENDLY layout (variable content mixed in prefix):")
    print("  messages = [")
    print('    {"role": "system",  "content": f"Today is {date}. You are..."},  # changes daily!')
    print('    {"role": "user",    "content": user_question},')
    print("  ]")
    print()
    print("  Rule: put everything that does NOT change at the top.")
    print("  Even a single token difference in the prefix busts the cache.")


def demo_estimate_savings() -> None:
    """DEMO 3: Estimating cache savings with shared.tokens.estimate_cost()."""
    client = get_client()
    messages = [
        {"role": "system", "content": STABLE_SYSTEM_PROMPT},
        {"role": "user", "content": "Review this code snippet for issues."},
    ]

    resp = client.chat.completions.create(model=MODEL, messages=messages)
    usage = resp.usage
    cached = usage.prompt_tokens_details.cached_tokens
    non_cached = usage.prompt_tokens - cached
    completion = usage.completion_tokens

    # Full cost without caching
    full_cost = estimate_cost(usage.prompt_tokens, completion, MODEL)
    # Cost with caching (cached tokens at 50%)
    cached_cost = estimate_cost(non_cached, completion, MODEL) + estimate_cost(
        cached, 0, MODEL
    ) * 0.5

    print(f"Prompt tokens  : {usage.prompt_tokens}")
    print(f"  cached       : {cached}")
    print(f"  non-cached   : {non_cached}")
    print(f"Completion     : {completion}")
    print()
    print(f"Full cost (no cache) : {format_cost(full_cost)}")
    print(f"Cost with cache      : {format_cost(cached_cost)}")
    if full_cost > 0:
        savings_pct = (full_cost - cached_cost) / full_cost * 100
        print(f"Savings              : {savings_pct:.1f}%")


def anti_pattern_variable_prefix() -> None:
    """ANTI-PATTERN 1: Variable content at the start of the prompt defeats caching.

    OpenAI's cache is keyed on the exact byte sequence of the prompt prefix.
    If anything before the stable content changes between calls, the cached
    prefix never matches — even if the stable part is identical.
    """
    from datetime import date

    today = date.today().isoformat()
    print("ANTI-PATTERN: Variable content in the prefix")
    print()
    print("  Each call with a fresh date string creates a NEW cache entry:")
    print(f'    system: "Current date: {today}. You are a helpful assistant."')
    print('    system: "Current date: 2025-01-01. You are a helpful assistant."')
    print("  -> These two prompts share 0 cached tokens despite identical stable content.")
    print()
    print("  Fix: move the date to the user message, not the system prompt.")
    print(f'    user: "Today is {today}. What is the exchange rate for EUR/USD?"')


def anti_pattern_not_checking_cached_tokens() -> None:
    """ANTI-PATTERN 2: Not checking cached_tokens when debugging high costs.

    If your per-call costs are higher than expected, cached_tokens=0 on
    every call is a strong signal that caching is not working. Check that:
      - Your stable prefix is >= 1024 tokens
      - You call the same model repeatedly within 5–10 minutes
      - The prefix bytes are identical (no dynamic timestamps/UUIDs in prefix)
    """
    print("ANTI-PATTERN: Ignoring usage.prompt_tokens_details.cached_tokens")
    print()
    print("  Symptom: costs are higher than expected on repeated calls.")
    print("  Diagnosis:")
    print("    resp = client.chat.completions.create(...)")
    print("    print(resp.usage.prompt_tokens_details.cached_tokens)  # 0 = cache miss!")
    print()
    print("  Checklist when cached_tokens == 0:")
    print("    [ ] Stable prefix >= 1024 tokens?")
    print("    [ ] Same exact bytes each call (no UUIDs/timestamps in prefix)?")
    print("    [ ] Within cache TTL (~5–10 min)?")
    print("    [ ] Model supports caching (gpt-4o, gpt-4o-mini — yes; o1 — yes)?")


if __name__ == "__main__":
    sep = "=" * 60
    mode = "MOCK" if is_mock() else "LIVE"
    print(f"{NL}{sep}{NL}Domain 12 - Task 12.9: Prompt Caching [{mode}]{NL}{sep}")

    print(f"{NL}--- DEMO 1: Reading cached_tokens ---")
    demo_reading_cached_tokens()

    print(f"{NL}--- DEMO 2: Cache-Friendly Prompt Layout ---")
    demo_cache_friendly_layout()

    print(f"{NL}--- DEMO 3: Estimating Savings ---")
    demo_estimate_savings()

    print(f"{NL}--- ANTI-PATTERN 1: Variable Content in Prefix ---")
    anti_pattern_variable_prefix()

    print(f"{NL}--- ANTI-PATTERN 2: Not Checking cached_tokens ---")
    anti_pattern_not_checking_cached_tokens()

    print(f"{NL}KEY TAKEAWAYS:")
    print("  1. Prefix caching kicks in automatically for prompts >= 1024 tokens")
    print("  2. cached_tokens are billed at 50% — check usage.prompt_tokens_details.cached_tokens")
    print("  3. Put stable content (system prompt, docs) FIRST; variable content LAST")
    print("  4. Cache TTL is ~5–10 min; warm cache gives ~50% cost reduction on input tokens")
    print("  5. cached_tokens=0 on every call signals a cache-busting prefix — investigate!")
