"""Domain 9 - Task 9.1: Batch Request Format

CONCEPTS:
  1. JSONL format — one JSON object per line, no trailing commas
  2. custom_id — unique string to match input to output
  3. method and url — always "POST" and "/v1/chat/completions"
  4. body — standard chat completion request body

Mnemonic: JCMB — Jsonl, Custom_id, Method_url, Body

Run:
  uv run python 01_batch_request_format.py
"""

NL = chr(10)
MODEL = "gpt-4o-mini"

import json

from shared.mock import is_mock


def build_batch_request(custom_id: str, prompt: str, model: str = MODEL) -> dict:
    """Build a single batch request object."""
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 100,
        },
    }


def build_batch_jsonl(prompts: list[str]) -> bytes:
    """Convert a list of prompts into a JSONL batch file."""
    lines = []
    for i, prompt in enumerate(prompts):
        req = build_batch_request(f"req-{i+1:03d}", prompt)
        lines.append(json.dumps(req))
    return NL.join(lines).encode("utf-8")


def demo_batch_format() -> None:
    """DEMO 1: Build and display a batch JSONL file."""
    prompts = [
        "What is the capital of France?",
        "What is 15 × 17?",
        "Translate 'hello' to Spanish.",
    ]
    jsonl_bytes = build_batch_jsonl(prompts)
    print(f"  JSONL batch file ({len(jsonl_bytes)} bytes, {len(prompts)} requests):")
    for line in jsonl_bytes.decode().split(NL):
        parsed = json.loads(line)
        print(f"    {parsed['custom_id']}: {parsed['body']['messages'][0]['content']!r}")


def demo_jsonl_structure() -> None:
    """DEMO 2: Show the full structure of one batch request."""
    req = build_batch_request("req-001", "Hello, world!")
    print("  One batch request:")
    for key, value in req.items():
        if key == "body":
            print("    body:")
            for bk, bv in value.items():
                print(f"      {bk}: {bv!r}")
        else:
            print(f"    {key}: {value!r}")


def anti_pattern_wrong_format() -> None:
    """ANTI-PATTERN 1: Common batch format mistakes."""
    print("  ANTI-PATTERN: using 'endpoint' instead of 'url'")
    wrong = {"custom_id": "req-1", "endpoint": "/v1/chat/completions"}
    print(f"    Wrong: {json.dumps(wrong)}")
    correct = {"custom_id": "req-1", "method": "POST", "url": "/v1/chat/completions", "body": {}}
    print(f"    Correct: {json.dumps(correct)}")
    print(f"{NL}  ANTI-PATTERN: JSON not JSONL (array instead of one obj per line)")
    print(f"    Wrong:   [{json.dumps({'a':1})}, {json.dumps({'b':2})}]")
    print(f"    Correct: {json.dumps({'a':1})}{NL}             {json.dumps({'b':2})}")


if __name__ == "__main__":
    sep = "=" * 60
    mode = "MOCK" if is_mock() else "LIVE"
    print(f"{NL}{sep}{NL}Domain 9 - Task 9.1: Batch Request Format [{mode}]{NL}{sep}")

    print(f"{NL}--- DEMO 1: Batch JSONL File ---")
    demo_batch_format()

    print(f"{NL}--- DEMO 2: Full Structure ---")
    demo_jsonl_structure()

    print(f"{NL}--- ANTI-PATTERN 1: Format Mistakes ---")
    anti_pattern_wrong_format()

    print(f"{NL}KEY TAKEAWAYS:")
    print("  1. JSONL = one JSON object per line, no commas between lines")
    print("  2. custom_id is your key to match batch output to input")
    print("  3. method must be 'POST'; url must be '/v1/chat/completions'")
    print("  4. body is a standard chat completions request — all params supported")
