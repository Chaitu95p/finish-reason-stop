"""Exercise 6 - Task 6.1: Prepare Fine-Tuning Dataset

GOAL: Build and validate a fine-tuning dataset in the required JSONL format.
Start from raw (input, output) pairs and produce upload-ready bytes.

SKILLS PRACTICED:
  - Fine-tuning JSONL format (messages with system/user/assistant)
  - Dataset validation before upload
  - Byte serialization for files.create()

Run:
  uv run python 01_prepare_dataset.py
"""

NL = chr(10)

import io
import json

from shared.mock import get_client, is_mock

SYSTEM_PROMPT = "Classify the sentiment. Reply with exactly one word: positive, negative, or neutral."

RAW_PAIRS: list[tuple[str, str]] = [
    ("This product exceeded my expectations!", "positive"),
    ("Broke after one day. Terrible quality.", "negative"),
    ("Works as described. Arrived on time.", "neutral"),
    ("Absolutely love it. Best purchase this year!", "positive"),
    ("Customer support was unhelpful and rude.", "negative"),
    ("Average product, nothing special.", "neutral"),
    ("Five stars! Would highly recommend.", "positive"),
    ("Missing parts in the box. Very disappointed.", "negative"),
    ("Does the job but nothing remarkable.", "neutral"),
    ("Exceptional build quality and fast shipping!", "positive"),
]


def build_fine_tuning_example(system: str, user: str, assistant: str) -> dict:
    """Build one fine-tuning training example."""
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]
    }


def build_dataset(pairs: list[tuple[str, str]], system: str) -> list[dict]:
    """Convert (input, output) pairs into fine-tuning examples."""
    return [build_fine_tuning_example(system, user, label) for user, label in pairs]


def validate_dataset(examples: list[dict]) -> list[str]:
    """Return list of error strings; empty means dataset is valid."""
    errors = []
    for i, ex in enumerate(examples):
        msgs = ex.get("messages", [])
        if not msgs:
            errors.append(f"Example {i}: no messages")
            continue
        roles = {m.get("role") for m in msgs}
        if "user" not in roles:
            errors.append(f"Example {i}: missing user message")
        if "assistant" not in roles:
            errors.append(f"Example {i}: missing assistant message")
        for j, m in enumerate(msgs):
            if not m.get("content", "").strip():
                errors.append(f"Example {i}, msg {j}: empty content")
    if len(examples) < 10:
        errors.append(f"Dataset has {len(examples)} examples; minimum is 10")
    return errors


def to_jsonl_bytes(examples: list[dict]) -> bytes:
    """Serialize dataset to JSONL bytes suitable for files.create()."""
    return NL.join(json.dumps(ex) for ex in examples).encode("utf-8")


if __name__ == "__main__":
    sep = "=" * 60
    mode = "MOCK" if is_mock() else "LIVE"
    print(f"{NL}{sep}{NL}Exercise 6 - Task 6.1: Prepare Dataset [{mode}]{NL}{sep}")

    dataset = build_dataset(RAW_PAIRS, SYSTEM_PROMPT)
    errors = validate_dataset(dataset)

    print(f"{NL}  Dataset size: {len(dataset)} examples")
    if errors:
        print("  Validation FAILED:")
        for e in errors:
            print(f"    {e}")
    else:
        print("  Validation PASSED")

    jsonl_bytes = to_jsonl_bytes(dataset)
    print(f"  JSONL size: {len(jsonl_bytes)} bytes")
    print(f"{NL}  First example:")
    for msg in dataset[0]["messages"]:
        print(f"    [{msg['role']}]: {msg['content']!r}")

    # Demonstrate upload (mock)
    client = get_client()
    f = io.BytesIO(jsonl_bytes)
    f.name = "training_data.jsonl"
    file_obj = client.files.create(file=f, purpose="fine-tune")  # type: ignore[attr-defined]
    print(f"{NL}  Uploaded file ID: {file_obj.id!r}")

    print(f"{NL}KEY TAKEAWAYS:")
    print("  1. Every example needs system + user + assistant messages in that order")
    print("  2. Validate before uploading — bad format causes silent job failures")
    print("  3. Consistent system prompt across ALL examples is critical")
