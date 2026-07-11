"""Domain 11 - Task 11.3: Timeout Handling

CONCEPTS:
  1. Global timeout — OpenAI(timeout=30.0) applies to all requests
  2. Per-request timeout — client.chat.completions.create(..., timeout=10.0)
  3. APITimeoutError — raised when timeout elapses; safe to retry
  4. Httpx Timeout object — fine-grained: connect, read, write, pool timeouts

Mnemonic: GPHA — Global, Per_request, Httpx_timeout, APITimeoutError

Run:
  uv run python 03_timeout_handling.py
"""

NL = chr(10)
MODEL = "gpt-4o"


from shared.mock import get_client, is_mock


def demo_global_timeout() -> None:
    """DEMO 1: Setting a global timeout on the client."""
    print("  Global timeout via client constructor:")
    print("""
    client = openai.OpenAI(
        timeout=30.0,   # seconds; applies to all requests
                        # default is httpx.Timeout(600.0) — way too long
    )
    """)
    print("  Recommended defaults:")
    print("    Interactive endpoints (chat): 30s")
    print("    Batch/long-running endpoints: 120-300s")
    print("    Embeddings (fast): 10-15s")


def demo_per_request_timeout() -> None:
    """DEMO 2: Override timeout per request."""
    client = get_client()
    print("  Per-request timeout override:")
    print("""
    response = client.chat.completions.create(
        model=\"gpt-4o\",
        messages=[{\"role\": \"user\", \"content\": \"Hello\"}],
        timeout=10.0,   # overrides the client-level timeout
    )
    """)

    # Demo: call with mock client (no actual timeout)
    response = client.chat.completions.create(  # type: ignore[attr-defined]
        model=MODEL,
        messages=[{"role": "user", "content": "Quick test"}],
    )
    print(f"  Response content: {response.choices[0].message.content!r}")


def demo_httpx_timeout_object() -> None:
    """DEMO 3: Fine-grained httpx.Timeout for different timeout phases."""
    print("  Fine-grained timeout with httpx.Timeout:")
    print("""
    import httpx

    client = openai.OpenAI(
        timeout=httpx.Timeout(
            connect=5.0,    # TCP connection establishment
            read=30.0,      # time to receive the first byte after headers sent
            write=5.0,      # time to send the request body
            pool=3.0,       # time to acquire a connection from the pool
        )
    )
    """)
    print("  Most useful: set read timeout longer than connect timeout")
    print("  Read is where streaming responses take time")


def demo_timeout_error_handling() -> None:
    """DEMO 4: Catching and handling APITimeoutError."""
    print("  Handling APITimeoutError:")
    print("""
    try:
        response = client.chat.completions.create(
            model=\"gpt-4o\",
            messages=[{\"role\": \"user\", \"content\": prompt}],
            timeout=15.0,
        )
    except openai.APITimeoutError:
        # Safe to retry — the request may or may not have reached the server
        # Use idempotency keys if retrying mutations (see 04_idempotency.py)
        log_timeout_event()
        raise  # let the retry decorator handle it
    """)
    print("  Note: APITimeoutError is a subclass of APIConnectionError")
    print("  You can catch both with: except openai.APIConnectionError")


def demo_streaming_timeout() -> None:
    """DEMO 5: Timeouts behave differently with streaming."""
    print("  Streaming timeout behavior:")
    print("    • timeout applies to receiving the FIRST chunk, not the full response")
    print("    • if the server starts streaming but is slow, no timeout is raised")
    print("    • use max_tokens to bound how long streaming can take")
    print("""
    # Safe streaming pattern with max_tokens to bound duration
    with client.chat.completions.create(
        model=\"gpt-4o\",
        messages=[...],
        stream=True,
        max_tokens=500,    # caps response length → caps streaming time
        timeout=30.0,      # covers time to first chunk
    ) as stream:
        for chunk in stream:
            delta = chunk.choices[0].delta.content or \"\"
            print(delta, end=\"\", flush=True)
    """)


if __name__ == "__main__":
    sep = "=" * 60
    mode = "MOCK" if is_mock() else "LIVE"
    print(f"{NL}{sep}{NL}Domain 11 - Task 11.3: Timeout Handling [{mode}]{NL}{sep}")

    print(f"{NL}--- DEMO 1: Global Timeout ---")
    demo_global_timeout()

    print(f"{NL}--- DEMO 2: Per-Request Timeout ---")
    demo_per_request_timeout()

    print(f"{NL}--- DEMO 3: Httpx Timeout Object ---")
    demo_httpx_timeout_object()

    print(f"{NL}--- DEMO 4: Error Handling ---")
    demo_timeout_error_handling()

    print(f"{NL}--- DEMO 5: Streaming Timeouts ---")
    demo_streaming_timeout()

    print(f"{NL}KEY TAKEAWAYS:")
    print("  1. Default SDK timeout is 600s — always set an explicit timeout")
    print("  2. Per-request timeout overrides client-level timeout; use for known-fast endpoints")
    print("  3. APITimeoutError is retryable — the request may not have reached the server")
    print("  4. With streaming, timeout governs time-to-first-chunk, not total response time")
