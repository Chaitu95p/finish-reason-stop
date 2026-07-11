"""Domain 11 - Task 11.4: Idempotency

CONCEPTS:
  1. Idempotency key — unique string that deduplicates retried requests
  2. X-Idempotency-Key header — passed as extra_headers to the SDK
  3. When to use — any retried mutating operation (uploads, fine-tune jobs)
  4. Key generation — UUID4 or hash of request content

Mnemonic: UHWG — Unique_key, Header_injection, When_mutations, Generate_UUID

Run:
  uv run python 04_idempotency.py
"""

NL = chr(10)
MODEL = "gpt-4o"

import hashlib
import uuid

from shared.mock import get_client, is_mock


def new_idempotency_key() -> str:
    """Generate a unique idempotency key."""
    return str(uuid.uuid4())


def content_based_key(content: str, model: str) -> str:
    """Generate a deterministic key based on request content."""
    raw = f"{model}:{content}"
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


def demo_idempotency_key_usage() -> None:
    """DEMO 1: Passing an idempotency key via extra_headers."""
    client = get_client()
    key = new_idempotency_key()
    print(f"  Generated key: {key!r}")
    print(f"{NL}  How to attach to a request:")
    print("""
    response = client.chat.completions.create(
        model=\"gpt-4o\",
        messages=[{\"role\": \"user\", \"content\": prompt}],
        extra_headers={\"X-Idempotency-Key\": idempotency_key},
    )
    """)
    # Mock call to show it works
    response = client.chat.completions.create(  # type: ignore[attr-defined]
        model=MODEL,
        messages=[{"role": "user", "content": "Hello"}],
        extra_headers={"X-Idempotency-Key": key},
    )
    print(f"  Response: {response.choices[0].message.content!r}")


def demo_content_based_key() -> None:
    """DEMO 2: Deterministic keys from request content."""
    prompt = "Summarize the history of Rome in one sentence."
    key = content_based_key(prompt, MODEL)
    print(f"  Prompt: {prompt!r}")
    print(f"  Deterministic key: {key!r}")
    print("  Same prompt always produces same key — server deduplicates repeated retries")
    key2 = content_based_key(prompt, MODEL)
    assert key == key2, "keys must match for same content"
    print(f"  Verified: second call produces same key: {key2 == key}")


def demo_when_to_use() -> None:
    """DEMO 3: When idempotency keys matter."""
    print("  USE idempotency keys for:")
    print("    ✓ File uploads (client.files.create) — prevent duplicate files")
    print("    ✓ Fine-tuning jobs — prevent duplicate training jobs on retry")
    print("    ✓ Any request you retry after a timeout or connection error")
    print(f"{NL}  NOT needed for:")
    print("    ✗ Read operations (retrieve, list) — already idempotent")
    print("    ✗ Chat completions normally — creating a new response is the intent")
    print("    ✗ Embeddings — cheap to regenerate, retry without key is fine")
    print(f"{NL}  Key validity window:")
    print("    • OpenAI caches idempotency keys for ~24 hours")
    print("    • Within 24h: same key = same response, no duplicate processing")
    print("    • After 24h: key can be reused safely")


def demo_retry_with_key() -> None:
    """DEMO 4: Full retry loop with idempotency key."""
    print("  Retry loop preserving idempotency key:")
    print("""
    import openai, uuid, time

    key = str(uuid.uuid4())  # generate ONCE before the loop
    max_retries = 3

    for attempt in range(max_retries):
        try:
            file_obj = client.files.create(
                file=open(\"training_data.jsonl\", \"rb\"),
                purpose=\"fine-tune\",
                extra_headers={\"X-Idempotency-Key\": key},  # same key every attempt
            )
            break
        except openai.APITimeoutError:
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)
    """)
    print("  Critical: generate the key OUTSIDE the retry loop")
    print("  If you generate inside the loop, each attempt has a different key → no dedup")


if __name__ == "__main__":
    sep = "=" * 60
    mode = "MOCK" if is_mock() else "LIVE"
    print(f"{NL}{sep}{NL}Domain 11 - Task 11.4: Idempotency [{mode}]{NL}{sep}")

    print(f"{NL}--- DEMO 1: Key Usage ---")
    demo_idempotency_key_usage()

    print(f"{NL}--- DEMO 2: Content-Based Keys ---")
    demo_content_based_key()

    print(f"{NL}--- DEMO 3: When to Use ---")
    demo_when_to_use()

    print(f"{NL}--- DEMO 4: Retry Loop ---")
    demo_retry_with_key()

    print(f"{NL}KEY TAKEAWAYS:")
    print("  1. Pass idempotency key via extra_headers={'X-Idempotency-Key': key}")
    print("  2. Generate the key ONCE before the retry loop, not inside it")
    print("  3. Use UUID4 for unique keys; use content hash for deterministic dedup")
    print("  4. Critical for file uploads and fine-tuning jobs — not needed for chat")
