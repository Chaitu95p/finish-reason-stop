"""Domain 11 - Task 11.2: Retry Patterns

CONCEPTS:
  1. SDK built-in retry — OpenAI(max_retries=N) handles 429 and 5xx automatically
  2. Custom exponential backoff — when you need control over what to retry
  3. Jitter — avoid thundering herd: sleep(base * 2**attempt + random.uniform(0, 1))
  4. Max delay cap — don't wait longer than 60s even with exponential growth

Mnemonic: BJMC — Backoff, Jitter, Max_cap, Client_builtin

Run:
  uv run python 02_retry_patterns.py
"""

NL = chr(10)
MODEL = "gpt-4o"

import random
import time

import openai
from shared.mock import get_client, is_mock


def demo_builtin_retry() -> None:
    """DEMO 1: SDK built-in retry via max_retries parameter."""
    print("  SDK built-in retry (recommended for most cases):")
    print("""
    # Retries 429 and 5xx automatically with exponential backoff
    client = openai.OpenAI(
        max_retries=3,      # default is 2
        timeout=30.0,       # default is 600s (too long for most apps)
    )
    """)
    print("  What it retries: RateLimitError (429), InternalServerError (500/502/503/504)")
    print("  What it does NOT retry: AuthenticationError, BadRequestError, NotFoundError")


def demo_custom_backoff() -> None:
    """DEMO 2: Custom exponential backoff decorator from shared.retry."""
    print("  Custom backoff from shared.retry:")
    print("""
    @exponential_backoff(max_retries=4, base_delay=1.0, max_delay=60.0)
    def call_api() -> str:
        response = client.chat.completions.create(...)
        return response.choices[0].message.content
    """)
    print("  Delay sequence (with jitter): 1s, 2s, 4s, 8s, ... capped at 60s")


def demo_manual_backoff_loop() -> None:
    """DEMO 3: Manual retry loop with jitter — explicit, readable."""
    client = get_client()
    max_retries = 3
    base_delay = 1.0
    max_delay = 60.0
    last_error: Exception | None = None

    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(  # type: ignore[attr-defined]
                model=MODEL,
                messages=[{"role": "user", "content": "Hello, how are you?"}],
            )
            content = response.choices[0].message.content or ""
            print(f"  Attempt {attempt + 1}: SUCCESS — {content!r}")
            return
        except (openai.RateLimitError, openai.APIConnectionError, openai.APITimeoutError) as e:
            last_error = e
            if attempt >= max_retries:
                break
            delay = min(base_delay * (2 ** attempt) + random.uniform(0, 1), max_delay)
            print(f"  Attempt {attempt + 1}: RETRY after {delay:.2f}s ({type(e).__name__})")
            time.sleep(delay * 0.01)  # scaled down for demo speed
        except openai.APIStatusError as e:
            if e.status_code in {500, 502, 503, 504}:
                last_error = e
                if attempt >= max_retries:
                    break
                delay = min(base_delay * (2 ** attempt) + random.uniform(0, 1), max_delay)
                print(f"  Attempt {attempt + 1}: RETRY after {delay:.2f}s (status {e.status_code})")
                time.sleep(delay * 0.01)
            else:
                print(f"  Attempt {attempt + 1}: NON-RETRYABLE {type(e).__name__} (status {e.status_code})")
                return

    print(f"  All retries exhausted: {type(last_error).__name__ if last_error else 'unknown'}")


def demo_retry_after_header() -> None:
    """DEMO 4: Respecting Retry-After header on 429 responses."""
    print("  Respecting Retry-After header from 429 responses:")
    print("""
    except openai.RateLimitError as e:
        retry_after = e.response.headers.get("Retry-After")
        if retry_after:
            wait = float(retry_after)
        else:
            wait = min(base_delay * (2 ** attempt), max_delay)
        time.sleep(wait)
    """)
    print("  The SDK's built-in retry already reads Retry-After automatically.")
    print("  Only implement this manually if you need custom behavior.")


if __name__ == "__main__":
    sep = "=" * 60
    mode = "MOCK" if is_mock() else "LIVE"
    print(f"{NL}{sep}{NL}Domain 11 - Task 11.2: Retry Patterns [{mode}]{NL}{sep}")

    print(f"{NL}--- DEMO 1: Built-in SDK Retry ---")
    demo_builtin_retry()

    print(f"{NL}--- DEMO 2: Custom Backoff Decorator ---")
    demo_custom_backoff()

    print(f"{NL}--- DEMO 3: Manual Retry Loop ---")
    demo_manual_backoff_loop()

    print(f"{NL}--- DEMO 4: Retry-After Header ---")
    demo_retry_after_header()

    print(f"{NL}KEY TAKEAWAYS:")
    print("  1. OpenAI(max_retries=3) is the easiest correct approach — use it first")
    print("  2. Add jitter to avoid thundering herd: sleep(base * 2**n + random(0,1))")
    print("  3. Never retry on 400/401/403/404 — these will fail again immediately")
    print("  4. Cap maximum delay at 60s — longer waits degrade user experience")
