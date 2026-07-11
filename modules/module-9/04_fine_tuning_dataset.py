"""Domain 9 - Task 9.4: Fine-Tuning Dataset Preparation

CONCEPTS:
  1. JSONL format — system/user/assistant triples, one per line
  2. Dataset quality — consistent system prompt, diverse examples
  3. Minimum 10 examples — recommended 50-100 for reliable results
  4. Validation — check format before uploading

Mnemonic: QVUM — Quality, Variety, Uniform_format, Minimum_count

Run:
  uv run python 04_fine_tuning_dataset.py
"""

NL = chr(10)
MODEL = "gpt-4o-mini"

import json

from shared.mock import is_mock


def make_example(system: str, user: str, assistant: str) -> dict:
    """Build a single fine-tuning training example."""
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]
    }


def build_sentiment_dataset() -> list[dict]:
    """Build a small sentiment classification dataset."""
    system = "Classify text sentiment. Reply with exactly one word: positive, negative, or neutral."
    examples = [
        ("I absolutely love this product!", "positive"),
        ("This is the worst purchase I've ever made.", "negative"),
        ("The package arrived on time.", "neutral"),
        ("Amazing quality, highly recommend!", "positive"),
        ("Completely disappointed with the service.", "negative"),
        ("Product works as described.", "neutral"),
        ("Best thing I bought this year!", "positive"),
        ("Broken on arrival. Terrible.", "negative"),
        ("Nothing special, does the job.", "neutral"),
        ("Outstanding customer support!", "positive"),
        ("Waited 3 weeks and still no delivery.", "negative"),
        ("Average quality for the price.", "neutral"),
    ]
    return [make_example(system, user, label) for user, label in examples]


def validate_dataset(examples: list[dict]) -> list[str]:
    """Validate dataset format. Returns list of error strings."""
    errors = []
    for i, ex in enumerate(examples):
        messages = ex.get("messages", [])
        if not messages:
            errors.append(f"Example {i}: missing 'messages' key")
            continue
        roles = [m.get("role") for m in messages]
        if "user" not in roles:
            errors.append(f"Example {i}: no user message")
        if "assistant" not in roles:
            errors.append(f"Example {i}: no assistant message")
        for j, msg in enumerate(messages):
            if not msg.get("content"):
                errors.append(f"Example {i}, message {j}: empty content")
    return errors


def to_jsonl(examples: list[dict]) -> bytes:
    """Serialize examples to JSONL bytes for upload."""
    return NL.join(json.dumps(ex) for ex in examples).encode("utf-8")


def demo_dataset_format() -> None:
    """DEMO 1: Show the required JSONL format."""
    dataset = build_sentiment_dataset()
    print(f"  Built {len(dataset)} training examples")
    print(f"{NL}  Example 0 (formatted):")
    ex = dataset[0]
    for msg in ex["messages"]:
        print(f"    [{msg['role']}]: {msg['content']!r}")
    print(f"{NL}  As JSONL line:")
    print(f"    {json.dumps(ex)}")


def demo_validation() -> None:
    """DEMO 2: Dataset validation before upload."""
    dataset = build_sentiment_dataset()
    errors = validate_dataset(dataset)
    if errors:
        print(f"  Validation FAILED — {len(errors)} errors:")
        for e in errors:
            print(f"    {e}")
    else:
        print(f"  Validation PASSED — {len(dataset)} examples OK")

    # Show a broken example
    bad_dataset = [{"messages": [{"role": "user", "content": "hi"}]}]  # missing assistant
    bad_errors = validate_dataset(bad_dataset)
    print(f"{NL}  Bad example errors:")
    for e in bad_errors:
        print(f"    {e}")


def demo_dataset_quality_checklist() -> None:
    """DEMO 3: Quality guidelines for effective fine-tuning."""
    print("  Dataset quality checklist:")
    print("    ✓ 50-100+ examples (10 is minimum, not optimal)")
    print("    ✓ Consistent system prompt across ALL examples")
    print("    ✓ Diverse inputs — cover edge cases and variations")
    print("    ✓ High-quality assistant responses — these are the targets")
    print("    ✓ Balanced distribution across classes/task types")
    print(f"{NL}  Common mistakes:")
    print("    ✗ System prompt varies per example")
    print("    ✗ All examples from same source (no diversity)")
    print("    ✗ Low-quality or incorrect assistant responses")
    print("    ✗ Only happy-path examples (no edge cases)")


if __name__ == "__main__":
    sep = "=" * 60
    mode = "MOCK" if is_mock() else "LIVE"
    print(f"{NL}{sep}{NL}Domain 9 - Task 9.4: Fine-Tuning Dataset [{mode}]{NL}{sep}")

    print(f"{NL}--- DEMO 1: Dataset Format ---")
    demo_dataset_format()

    print(f"{NL}--- DEMO 2: Validation ---")
    demo_validation()

    print(f"{NL}--- DEMO 3: Quality Checklist ---")
    demo_dataset_quality_checklist()

    print(f"{NL}KEY TAKEAWAYS:")
    print("  1. Fine-tuning JSONL = one {messages:[...]} object per line")
    print("  2. Always validate format before uploading — bad format fails silently")
    print("  3. Consistent system prompt is critical — it becomes the model's persona")
    print("  4. Quality > Quantity: 50 excellent examples beat 500 mediocre ones")
