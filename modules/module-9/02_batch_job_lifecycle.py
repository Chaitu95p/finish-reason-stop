"""Domain 9 - Task 9.2: Batch Job Lifecycle

CONCEPTS:
  1. client.batches.create() — submit a batch job
  2. Status transitions: validating → in_progress → completed (or failed)
  3. completion_window="24h" — required param; up to 24h to complete
  4. output_file_id — download results when status == "completed"

Mnemonic: VSCO — Validating, Status_poll, Completed, Output_file

Run:
  uv run python 02_batch_job_lifecycle.py
"""

NL = chr(10)
MODEL = "gpt-4o-mini"

import io
import json
import time

from shared.mock import get_client, is_mock


def upload_batch_file(client: object, jsonl_bytes: bytes) -> str:
    """Upload JSONL bytes and return file ID."""
    f = io.BytesIO(jsonl_bytes)
    f.name = "batch_requests.jsonl"
    file_obj = client.files.create(file=f, purpose="batch")  # type: ignore[attr-defined]
    return file_obj.id


def poll_batch_status(client: object, batch_id: str, max_polls: int = 5) -> str:
    """Poll batch status until completed or failed."""
    statuses = ["validating", "in_progress", "finalizing", "completed"]
    for poll in range(max_polls):
        batch = client.batches.retrieve(batch_id)  # type: ignore[attr-defined]
        status = batch.status
        print(f"    Poll {poll+1}: status={status!r}")
        if status in ("completed", "failed", "cancelled", "expired"):
            return status
        time.sleep(0.01)  # real poll: 30-60 seconds
    return "timeout"


def demo_batch_lifecycle() -> None:
    """DEMO 1: Full batch job lifecycle."""
    client = get_client()

    # Step 1: Build JSONL batch file
    requests = [
        {"custom_id": "req-1", "method": "POST", "url": "/v1/chat/completions",
         "body": {"model": MODEL, "messages": [{"role": "user", "content": "What is 2+2?"}]}},
        {"custom_id": "req-2", "method": "POST", "url": "/v1/chat/completions",
         "body": {"model": MODEL, "messages": [{"role": "user", "content": "Capital of Japan?"}]}},
    ]
    jsonl = NL.join(json.dumps(r) for r in requests).encode()
    print(f"  Step 1: Built JSONL ({len(jsonl)} bytes, {len(requests)} requests)")

    # Step 2: Upload file
    file_id = upload_batch_file(client, jsonl)
    print(f"  Step 2: Uploaded file → {file_id!r}")

    # Step 3: Create batch
    batch = client.batches.create(
        input_file_id=file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )
    print(f"  Step 3: Created batch → id={batch.id!r} status={batch.status!r}")

    # Step 4: Poll status
    print("  Step 4: Polling status...")
    final_status = poll_batch_status(client, batch.id)
    print(f"  Step 4: Final status={final_status!r}")

    # Step 5: Download results (mock)
    if final_status == "completed":
        final_batch = client.batches.retrieve(batch.id)
        output_file_id = final_batch.output_file_id
        print(f"  Step 5: Output file ID={output_file_id!r}")
        print("  Step 5: Would call client.files.content(output_file_id) to download JSONL")


def demo_status_machine() -> None:
    """DEMO 2: All possible batch statuses."""
    statuses = {
        "validating": "Checking format and content of the request file",
        "failed": "Validation failed — check error_file_id for details",
        "in_progress": "Requests are being processed",
        "finalizing": "Processing done, preparing output file",
        "completed": "All requests done — output_file_id available",
        "expired": "Not completed within completion_window",
        "cancelling": "Cancellation requested",
        "cancelled": "Cancelled by user",
    }
    print("  Batch status state machine:")
    for status, desc in statuses.items():
        print(f"    {status:<15}: {desc}")


if __name__ == "__main__":
    sep = "=" * 60
    mode = "MOCK" if is_mock() else "LIVE"
    print(f"{NL}{sep}{NL}Domain 9 - Task 9.2: Batch Job Lifecycle [{mode}]{NL}{sep}")

    print(f"{NL}--- DEMO 1: Full Lifecycle ---")
    demo_batch_lifecycle()

    print(f"{NL}--- DEMO 2: Status Machine ---")
    demo_status_machine()

    print(f"{NL}KEY TAKEAWAYS:")
    print("  1. Upload JSONL → create batch → poll status → download output")
    print("  2. Poll every 30-60 seconds in production (not tight loop)")
    print("  3. Check output_file_id only when status == 'completed'")
    print("  4. Check error_file_id when status == 'failed' for per-request errors")
