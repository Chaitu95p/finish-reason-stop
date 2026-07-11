"""Exercise 6 - Task 6.2: Train and Evaluate Fine-Tuned Model

GOAL: Upload dataset → create fine-tuning job → poll status → evaluate
against base model on a held-out test set.

SKILLS PRACTICED:
  - fine_tuning.jobs.create() full lifecycle
  - A/B evaluation: base model vs fine-tuned model
  - Accuracy metric on structured output tasks

Run:
  uv run python 02_train_and_evaluate.py
"""

NL = chr(10)
BASE_MODEL = "gpt-4o-mini"
FINE_TUNED_MODEL = "ft:gpt-4o-mini-2024-07-18:my-org:sentiment:AbCd1234"

import io
import json
import time

from shared.mock import get_client, is_mock

SYSTEM_PROMPT = "Classify the sentiment. Reply with exactly one word: positive, negative, or neutral."

TRAINING_DATA: list[tuple[str, str]] = [
    ("Amazing product!", "positive"),
    ("Terrible experience.", "negative"),
    ("Works fine.", "neutral"),
    ("Love it!", "positive"),
    ("Completely broken.", "negative"),
    ("Nothing special.", "neutral"),
    ("Outstanding quality!", "positive"),
    ("Very disappointed.", "negative"),
    ("Decent product.", "neutral"),
    ("Highly recommend!", "positive"),
]

TEST_SET: list[tuple[str, str]] = [
    ("Great value for money!", "positive"),
    ("Would not buy again.", "negative"),
    ("It arrived on time.", "neutral"),
    ("Best product ever!", "positive"),
]


def make_jsonl(pairs: list[tuple[str, str]]) -> bytes:
    examples = [
        {"messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
            {"role": "assistant", "content": label},
        ]}
        for text, label in pairs
    ]
    return NL.join(json.dumps(ex) for ex in examples).encode()


def upload_and_train(client: object) -> str:
    """Upload dataset and start fine-tuning job. Returns job ID."""
    f = io.BytesIO(make_jsonl(TRAINING_DATA))
    f.name = "train.jsonl"
    file_obj = client.files.create(file=f, purpose="fine-tune")  # type: ignore[attr-defined]
    print(f"  Uploaded training file: {file_obj.id!r}")

    job = client.fine_tuning.jobs.create(  # type: ignore[attr-defined]
        training_file=file_obj.id,
        model=BASE_MODEL,
        suffix="sentiment",
    )
    print(f"  Created job: {job.id!r} status={job.status!r}")
    return job.id


def poll_until_done(client: object, job_id: str, max_polls: int = 5) -> str:
    """Poll job status until succeeded/failed."""
    for i in range(max_polls):
        job = client.fine_tuning.jobs.retrieve(job_id)  # type: ignore[attr-defined]
        print(f"  Poll {i+1}: status={job.status!r}")
        if job.status in ("succeeded", "failed", "cancelled"):
            if hasattr(job, "fine_tuned_model") and job.fine_tuned_model:
                print(f"  fine_tuned_model: {job.fine_tuned_model!r}")
            return job.status
        time.sleep(0.01)
    return "timeout"


def evaluate_model(client: object, model: str, test_set: list[tuple[str, str]]) -> float:
    """Evaluate model accuracy on test set."""
    correct = 0
    for text, expected in test_set:
        response = client.chat.completions.create(  # type: ignore[attr-defined]
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
            max_tokens=5,
        )
        predicted = (response.choices[0].message.content or "").strip().lower()
        if predicted == expected:
            correct += 1
    return correct / len(test_set)


if __name__ == "__main__":
    sep = "=" * 60
    mode = "MOCK" if is_mock() else "LIVE"
    print(f"{NL}{sep}{NL}Exercise 6 - Task 6.2: Train & Evaluate [{mode}]{NL}{sep}")

    client = get_client()

    print(f"{NL}  Step 1: Train")
    job_id = upload_and_train(client)

    print(f"{NL}  Step 2: Poll")
    final_status = poll_until_done(client, job_id)
    print(f"  Final status: {final_status!r}")

    print(f"{NL}  Step 3: Evaluate (mock — same results in mock mode)")
    base_accuracy = evaluate_model(client, BASE_MODEL, TEST_SET)
    ft_accuracy = evaluate_model(client, FINE_TUNED_MODEL, TEST_SET)
    print(f"  Base model accuracy:       {base_accuracy:.0%}")
    print(f"  Fine-tuned model accuracy: {ft_accuracy:.0%}")

    print(f"{NL}KEY TAKEAWAYS:")
    print("  1. upload → create job → poll status → evaluate on held-out test set")
    print("  2. Always compare fine-tuned model against base model on same test set")
    print("  3. In mock mode both models return the same response — use live API for real evaluation")
