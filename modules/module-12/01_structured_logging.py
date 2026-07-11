"""Domain 12 - Task 12.1: Structured Logging

CONCEPTS:
  1. JSON log lines — machine-parseable, grep-friendly
  2. request_id — trace a single API call through logs
  3. latency — measure time from send to first token / full response
  4. Log fields: model, tokens, cost, finish_reason, request_id, latency_ms

Mnemonic: JRLT — Json_lines, Request_id, Latency, Token_cost

Run:
  uv run python 01_structured_logging.py
"""

NL = chr(10)
MODEL = "gpt-4o"

import time
from typing import Any

from shared.logging import APICallLogger, new_request_id, structured_log
from shared.mock import get_client, is_mock
from shared.tokens import estimate_cost


def log_api_call(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    finish_reason: str,
    latency_ms: float,
    request_id: str,
) -> None:
    """Emit a structured log entry for an API call."""
    cost = estimate_cost(prompt_tokens, completion_tokens, model)
    structured_log(
        "openai_api_call",
        request_id=request_id,
        model=model,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        finish_reason=finish_reason,
        latency_ms=round(latency_ms, 1),
        cost_usd=round(cost, 6),
    )


def demo_structured_logging() -> None:
    """DEMO 1: Emit a structured log entry after each API call."""
    client = get_client()
    request_id = new_request_id()
    t0 = time.monotonic()
    response = client.chat.completions.create(  # type: ignore[attr-defined]
        model=MODEL,
        messages=[{"role": "user", "content": "Summarize quantum computing in one line."}],
    )
    latency_ms = (time.monotonic() - t0) * 1000
    usage = response.usage
    log_api_call(
        model=MODEL,
        prompt_tokens=getattr(usage, "prompt_tokens", 15),
        completion_tokens=getattr(usage, "completion_tokens", 30),
        finish_reason=response.choices[0].finish_reason,
        latency_ms=latency_ms,
        request_id=request_id,
    )
    print(f"  Response: {response.choices[0].message.content!r}")


def demo_api_call_logger() -> None:
    """DEMO 2: APICallLogger context manager."""
    client = get_client()
    prompt = "What is the speed of light?"
    with APICallLogger(operation=f"chat/{MODEL}") as logger:
        response = client.chat.completions.create(  # type: ignore[attr-defined]
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
    print(f"  Response: {response.choices[0].message.content!r}")


def demo_log_fields_reference() -> None:
    """DEMO 3: Reference log field schema."""
    sample_entry: dict[str, Any] = {
        "timestamp": "2024-01-15T10:30:00.123Z",
        "event": "openai_api_call",
        "request_id": "req_a1b2c3d4",
        "model": "gpt-4o",
        "prompt_tokens": 150,
        "completion_tokens": 42,
        "finish_reason": "stop",
        "latency_ms": 823.4,
        "cost_usd": 0.000567,
        "environment": "production",
    }
    print("  Standard log entry schema:")
    for k, v in sample_entry.items():
        print(f"    {k}: {v!r}")


def demo_anti_patterns() -> None:
    """DEMO 4: Logging anti-patterns."""
    print("  ANTI-PATTERN 1: Logging free-form strings instead of JSON")
    print("    print(f'API call took {latency:.2f}s')  ← ungrep-able, no structure")
    print(f"{NL}  ANTI-PATTERN 2: Logging full prompt content in production")
    print("    → PII leaks; log prompt_hash or first 50 chars instead")
    print(f"{NL}  ANTI-PATTERN 3: No request_id correlation")
    print("    → Can't trace a single user interaction through logs")


if __name__ == "__main__":
    sep = "=" * 60
    mode = "MOCK" if is_mock() else "LIVE"
    print(f"{NL}{sep}{NL}Domain 12 - Task 12.1: Structured Logging [{mode}]{NL}{sep}")

    print(f"{NL}--- DEMO 1: Structured Log Entry ---")
    demo_structured_logging()

    print(f"{NL}--- DEMO 2: APICallLogger Context Manager ---")
    demo_api_call_logger()

    print(f"{NL}--- DEMO 3: Log Field Reference ---")
    demo_log_fields_reference()

    print(f"{NL}--- DEMO 4: Anti-Patterns ---")
    demo_anti_patterns()

    print(f"{NL}KEY TAKEAWAYS:")
    print("  1. Log JSON lines, not free-form strings — structured logs are grep-able")
    print("  2. Every API call logs: request_id, model, tokens, cost, latency, finish_reason")
    print("  3. Use a correlation ID (request_id) to trace one user turn through logs")
    print("  4. Never log raw prompt content in production — log a hash or truncated preview")
