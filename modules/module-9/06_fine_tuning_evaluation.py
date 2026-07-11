"""Domain 9 - Task 9.6: Fine-Tuning Evaluation

CONCEPTS:
  1. A/B comparison — base model vs fine-tuned on same prompts
  2. Format compliance — does the model output match the target format?
  3. Cost comparison — fine-tuned inference costs differ from base
  4. When NOT to fine-tune — try prompting first

Mnemonic: ABFC — AB_compare, Format_check, Cost_delta, Compare_first

Run:
  uv run python 06_fine_tuning_evaluation.py
"""

NL = chr(10)
BASE_MODEL = "gpt-4o-mini"
FINE_TUNED_MODEL = "ft:gpt-4o-mini-2024-07-18:my-org:sentiment-v1:AbCdEfGh"

from shared.mock import get_client, is_mock

TEST_CASES = [
    ("I love this product, absolutely amazing!", "positive"),
    ("Worst experience ever. Never buying again.", "negative"),
    ("It arrived on time and works fine.", "neutral"),
    ("The build quality is exceptional.", "positive"),
    ("Three weeks late and broken on arrival.", "negative"),
]


def classify_sentiment(client: object, model: str, text: str) -> str:
    """Classify sentiment using specified model."""
    system = "Classify text sentiment. Reply with exactly one word: positive, negative, or neutral."
    response = client.chat.completions.create(  # type: ignore[attr-defined]
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": text},
        ],
        max_tokens=5,
    )
    return (response.choices[0].message.content or "").strip().lower()


def run_evaluation(client: object, model: str) -> dict:
    """Run all test cases and compute accuracy."""
    correct = 0
    results = []
    for text, expected in TEST_CASES:
        predicted = classify_sentiment(client, model, text)
        is_correct = predicted == expected
        correct += int(is_correct)
        results.append({"text": text[:40], "expected": expected, "predicted": predicted, "correct": is_correct})
    accuracy = correct / len(TEST_CASES)
    return {"accuracy": accuracy, "correct": correct, "total": len(TEST_CASES), "results": results}


def demo_ab_comparison() -> None:
    """DEMO 1: A/B comparison of base vs fine-tuned model."""
    client = get_client()
    print(f"  Test cases: {len(TEST_CASES)}")
    for model_label, model_id in [("Base model", BASE_MODEL), ("Fine-tuned", FINE_TUNED_MODEL)]:
        report = run_evaluation(client, model_id)
        print(f"{NL}  [{model_label}] ({model_id[:30]}...)")
        print(f"    Accuracy: {report['accuracy']:.0%} ({report['correct']}/{report['total']})")
        for r in report["results"]:
            icon = "✓" if r["correct"] else "✗"
            print(f"    {icon} {r['text']!r:42} → {r['predicted']!r} (expected {r['expected']!r})")


def demo_cost_comparison() -> None:
    """DEMO 2: Cost comparison — fine-tuned inference is more expensive."""
    print("  Fine-tuned model pricing (gpt-4o-mini, approximate):")
    print("    Base model input:        $0.150 / 1M tokens")
    print("    Fine-tuned input:        $0.300 / 1M tokens (2x)")
    print("    Base model output:       $0.600 / 1M tokens")
    print("    Fine-tuned output:       $1.200 / 1M tokens (2x)")
    print(f"{NL}  Fine-tuning training cost:")
    print("    Training tokens:         $0.050 / 1M tokens (one-time)")
    print(f"{NL}  Break-even analysis:")
    print("    Fine-tune if: consistent format compliance saves more than 2x inference cost")
    print("    Alternative: try a larger base model first (gpt-4o vs gpt-4o-mini)")


def demo_when_to_fine_tune() -> None:
    """DEMO 3: Decision framework for fine-tuning."""
    print("  Fine-tune ONLY if you've exhausted these first:")
    steps = [
        "1. Prompt engineering — few-shot examples in the system prompt",
        "2. JSON mode or structured outputs — for format compliance",
        "3. Larger base model — gpt-4o instead of gpt-4o-mini",
        "4. Retrieval augmentation — provide context instead of baking knowledge",
    ]
    for step in steps:
        print(f"    {step}")
    print(f"{NL}  Fine-tuning IS the right choice when:")
    print("    ✓ You need consistent output format across thousands of calls")
    print("    ✓ Specific tone/persona/style that prompting can't reliably achieve")
    print("    ✓ Domain-specific knowledge absent from base model")
    print("    ✓ Latency reduction — shorter system prompts once baked in")


if __name__ == "__main__":
    sep = "=" * 60
    mode = "MOCK" if is_mock() else "LIVE"
    print(f"{NL}{sep}{NL}Domain 9 - Task 9.6: Fine-Tuning Evaluation [{mode}]{NL}{sep}")

    print(f"{NL}--- DEMO 1: A/B Comparison ---")
    demo_ab_comparison()

    print(f"{NL}--- DEMO 2: Cost Comparison ---")
    demo_cost_comparison()

    print(f"{NL}--- DEMO 3: When to Fine-Tune ---")
    demo_when_to_fine_tune()

    print(f"{NL}KEY TAKEAWAYS:")
    print("  1. Always A/B compare on a held-out test set — not just training examples")
    print("  2. Fine-tuned inference costs ~2x base model — factor into ROI calculation")
    print("  3. Try prompt engineering and RAG before committing to fine-tuning")
    print("  4. Format compliance is the #1 reason fine-tuning outperforms prompting")
