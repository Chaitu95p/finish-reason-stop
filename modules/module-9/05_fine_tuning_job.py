"""Domain 9 - Task 9.5: Fine-Tuning Jobs

CONCEPTS:
  1. client.fine_tuning.jobs.create() — start a fine-tuning job
  2. training_file — file ID from client.files.create(purpose="fine-tune")
  3. Polling events — client.fine_tuning.jobs.list_events(job_id)
  4. fine_tuned_model — the ft:gpt-4o-mini:...:id model name returned on completion

Mnemonic: UPLE — Upload, Poll, Learn, Endpoint

Run:
  uv run python 05_fine_tuning_job.py
"""

NL = chr(10)
MODEL = "gpt-4o-mini"

import io
import json
import time

from shared.mock import get_client, is_mock


def prepare_training_file(client: object, jsonl_bytes: bytes) -> str:
    """Upload JSONL bytes for fine-tuning; return file ID."""
    f = io.BytesIO(jsonl_bytes)
    f.name = "training_data.jsonl"
    file_obj = client.files.create(file=f, purpose="fine-tune")  # type: ignore[attr-defined]
    return file_obj.id


def build_minimal_dataset() -> bytes:
    """Build a tiny valid fine-tuning dataset."""
    system = "You are a concise assistant. Answer in one sentence."
    pairs = [
        ("What is Python?", "Python is a high-level, general-purpose programming language."),
        ("What is a list?", "A list is an ordered, mutable sequence of elements in Python."),
        ("What is a dict?", "A dict is an unordered collection of key-value pairs in Python."),
    ]
    lines = []
    for user, assistant in pairs:
        ex = {"messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]}
        lines.append(json.dumps(ex))
    return NL.join(lines).encode()


def poll_job_events(client: object, job_id: str, max_polls: int = 3) -> None:
    """Poll fine-tuning job events until complete."""
    for i in range(max_polls):
        job = client.fine_tuning.jobs.retrieve(job_id)  # type: ignore[attr-defined]
        print(f"    Poll {i+1}: status={job.status!r}")
        if job.status in ("succeeded", "failed", "cancelled"):
            if job.status == "succeeded":
                print(f"    fine_tuned_model: {job.fine_tuned_model!r}")
            break
        time.sleep(0.01)  # real polling: 60+ seconds per poll


def demo_fine_tuning_job() -> None:
    """DEMO 1: Full fine-tuning job lifecycle."""
    client = get_client()

    # Step 1: Upload training data
    jsonl_bytes = build_minimal_dataset()
    file_id = prepare_training_file(client, jsonl_bytes)
    print(f"  Step 1: Uploaded training file → {file_id!r}")

    # Step 2: Create fine-tuning job
    job = client.fine_tuning.jobs.create(  # type: ignore[attr-defined]
        training_file=file_id,
        model=MODEL,
    )
    print(f"  Step 2: Created job → id={job.id!r} status={job.status!r}")

    # Step 3: Poll for completion
    print("  Step 3: Polling job...")
    poll_job_events(client, job.id)


def demo_job_params() -> None:
    """DEMO 2: Fine-tuning job optional parameters."""
    print("  client.fine_tuning.jobs.create() key params:")
    params = {
        "training_file": "file-abc123  # required: file ID from files.create(purpose='fine-tune')",
        "model": "gpt-4o-mini  # required: base model to fine-tune",
        "validation_file": "file-xyz789  # optional: held-out eval set",
        "hyperparameters.n_epochs": "3  # optional: default 'auto' (1-10)",
        "suffix": "my-classifier  # optional: appended to ft: model name",
    }
    for param, description in params.items():
        print(f"    {param}: {description}")


def demo_model_name_format() -> None:
    """DEMO 3: The fine_tuned_model name format."""
    example = "ft:gpt-4o-mini-2024-07-18:your-org:my-classifier:AbCdEfGh"
    parts = example.split(":")
    print(f"  Fine-tuned model name: {example!r}")
    print("    ft:               — prefix indicating fine-tuned model")
    print(f"    {parts[1]}  — base model version")
    print(f"    {parts[2]}         — your organization name")
    print(f"    {parts[3]}  — optional suffix you provided")
    print(f"    {parts[4]}      — unique job identifier")
    print(f"{NL}  Use this name exactly in model= when calling the API")


if __name__ == "__main__":
    sep = "=" * 60
    mode = "MOCK" if is_mock() else "LIVE"
    print(f"{NL}{sep}{NL}Domain 9 - Task 9.5: Fine-Tuning Jobs [{mode}]{NL}{sep}")

    print(f"{NL}--- DEMO 1: Full Lifecycle ---")
    demo_fine_tuning_job()

    print(f"{NL}--- DEMO 2: Job Parameters ---")
    demo_job_params()

    print(f"{NL}--- DEMO 3: Model Name Format ---")
    demo_model_name_format()

    print(f"{NL}KEY TAKEAWAYS:")
    print("  1. Upload file with purpose='fine-tune', then reference its ID in jobs.create()")
    print("  2. Poll job status every 60+ seconds — fine-tuning takes 10-60+ minutes")
    print("  3. fine_tuned_model is a ft:... string — use it exactly as the model= param")
    print("  4. Use suffix= to make the model name recognizable in the dashboard")
