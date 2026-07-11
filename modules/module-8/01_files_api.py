"""Domain 8 - Task 8.1: Files API

CONCEPTS:
  1. client.files.create(file, purpose) — upload file
  2. purpose options — "batch", "fine-tune", "assistants"
  3. client.files.list() — enumerate uploaded files
  4. client.files.delete() — cleanup

Mnemonic: CLUD — Create, List, Upload, Delete

Run:
  uv run python 01_files_api.py
"""

from __future__ import annotations

import io
from dataclasses import dataclass

from shared.mock import get_client, is_mock

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

NL = chr(10)
MODEL = "gpt-4o"


# ---------------------------------------------------------------------------
# Domain objects
# ---------------------------------------------------------------------------


@dataclass
class FileRecord:
    """Represents a file uploaded to the OpenAI Files API."""

    file_id: str
    filename: str
    purpose: str
    status: str
    size_bytes: int
    created_at: int


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def upload_file(client: object, content: bytes, filename: str, purpose: str) -> FileRecord:
    """Upload a file to the Files API and return a FileRecord."""
    file_tuple = (filename, io.BytesIO(content), "application/octet-stream")
    result = client.files.create(file=file_tuple, purpose=purpose)  # type: ignore[union-attr]
    return FileRecord(
        file_id=result.id,
        filename=result.filename,
        purpose=result.purpose,
        status=result.status,
        size_bytes=result.bytes,
        created_at=result.created_at,
    )


def list_files(client: object) -> list[FileRecord]:
    """List all files uploaded to the Files API."""
    response = client.files.list()  # type: ignore[union-attr]
    records: list[FileRecord] = []
    for f in response:
        records.append(FileRecord(
            file_id=f.id,
            filename=f.filename,
            purpose=f.purpose,
            status=f.status,
            size_bytes=f.bytes,
            created_at=f.created_at,
        ))
    return records


def retrieve_file(client: object, file_id: str) -> FileRecord:
    """Retrieve metadata for a single file by ID."""
    f = client.files.retrieve(file_id)  # type: ignore[union-attr]
    return FileRecord(
        file_id=f.id,
        filename=f.filename,
        purpose=f.purpose,
        status=f.status,
        size_bytes=f.bytes,
        created_at=f.created_at,
    )


def delete_file(client: object, file_id: str) -> bool:
    """Delete a file and return True if successful."""
    result = client.files.delete(file_id)  # type: ignore[union-attr]
    return bool(result.deleted)


def make_jsonl_content(records: list[dict]) -> bytes:  # type: ignore[type-arg]
    """Convert list of dicts to JSONL bytes for batch upload."""
    import json
    lines = [json.dumps(r) for r in records]
    return NL.join(lines).encode("utf-8")


def make_text_content(text: str) -> bytes:
    """Convert plain text to bytes for upload."""
    return text.encode("utf-8")


def print_file_record(record: FileRecord, label: str | None = None) -> None:
    """Print a FileRecord in a readable format."""
    if label:
        print(f"  [{label}]")
    print(f"  ID:         {record.file_id}")
    print(f"  Filename:   {record.filename}")
    print(f"  Purpose:    {record.purpose}")
    print(f"  Status:     {record.status}")
    print(f"  Size:       {record.size_bytes} bytes")
    print(f"  Created at: {record.created_at}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    client = get_client()
    sep = "=" * 60

    print(sep)
    print("Domain 8 - Task 8.1: Files API")
    print(f"Mode: {'MOCK' if is_mock() else 'LIVE'}")
    print(sep)

    # DEMO 1: Upload a batch JSONL file
    print(NL + "DEMO 1: Upload a batch JSONL file")
    print("-" * 40)
    batch_data = [
        {"custom_id": "req-001", "method": "POST", "url": "/v1/chat/completions",
         "body": {"model": MODEL, "messages": [{"role": "user", "content": "Say hello"}]}},
        {"custom_id": "req-002", "method": "POST", "url": "/v1/chat/completions",
         "body": {"model": MODEL, "messages": [{"role": "user", "content": "Say goodbye"}]}},
    ]
    jsonl_bytes = make_jsonl_content(batch_data)  # type: ignore[arg-type]
    print(f"  Prepared JSONL content: {len(jsonl_bytes)} bytes, {len(batch_data)} records")

    batch_file = upload_file(client, jsonl_bytes, "batch_requests.jsonl", purpose="batch")
    print("  Uploaded batch file:")
    print_file_record(batch_file, "batch")

    # DEMO 2: Upload a fine-tuning file
    print(NL + "DEMO 2: Upload a fine-tuning JSONL file")
    print("-" * 40)
    finetune_data = [
        {"messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "4"},
        ]},
        {"messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of France?"},
            {"role": "assistant", "content": "Paris"},
        ]},
    ]
    ft_bytes = make_jsonl_content(finetune_data)  # type: ignore[arg-type]
    ft_file = upload_file(client, ft_bytes, "finetune_data.jsonl", purpose="fine-tune")
    print("  Uploaded fine-tune file:")
    print_file_record(ft_file, "fine-tune")

    # DEMO 3: Upload an assistants file
    print(NL + "DEMO 3: Upload a plain text file for assistants")
    print("-" * 40)
    doc_text = (
        "Product Manual v1.0" + NL
        + "====================" + NL
        + "Installation: Run ./install.sh" + NL
        + "Configuration: Edit config.yaml" + NL
        + "Troubleshooting: Check logs in /var/log/app/"
    )
    doc_bytes = make_text_content(doc_text)
    doc_file = upload_file(client, doc_bytes, "product_manual.txt", purpose="assistants")
    print("  Uploaded assistants file:")
    print_file_record(doc_file, "assistants")

    # DEMO 4: List all uploaded files
    print(NL + "DEMO 4: List all uploaded files")
    print("-" * 40)
    all_files = list_files(client)
    print(f"  Total files found: {len(all_files)}")
    for idx, f in enumerate(all_files, 1):
        print(f"  File {idx}: {f.file_id} | {f.filename} | purpose={f.purpose}")

    # DEMO 5: Retrieve a single file by ID
    print(NL + "DEMO 5: Retrieve file metadata by ID")
    print("-" * 40)
    retrieved = retrieve_file(client, batch_file.file_id)
    print("  Retrieved file:")
    print_file_record(retrieved, "retrieved")

    # DEMO 6: Purpose options summary
    print(NL + "DEMO 6: Purpose options summary")
    print("-" * 40)
    purposes = {
        "batch":       "Used with the Batch API for async bulk requests",
        "fine-tune":   "Training data for fine-tuning a model",
        "assistants":  "Knowledge files for the Assistants API",
    }
    for purpose, desc in purposes.items():
        print(f"  {purpose:<14} — {desc}")

    # DEMO 7: Delete files (cleanup)
    print(NL + "DEMO 7: Delete files (cleanup lifecycle)")
    print("-" * 40)
    for record in [batch_file, ft_file, doc_file]:
        ok = delete_file(client, record.file_id)
        status_str = "deleted" if ok else "FAILED"
        print(f"  {record.file_id} ({record.filename}) — {status_str}")

    # ANTI-PATTERN 1: Do not import openai.OpenAI directly for normal usage
    print(NL + "ANTI-PATTERN 1: Never import openai.OpenAI directly")
    print("-" * 40)
    print("  BAD:  from openai import OpenAI; client = OpenAI()")
    print("  GOOD: from shared.mock import get_client; client = get_client()")
    print("  Reason: Direct import bypasses mock — scripts fail without API key.")

    print(NL + sep)
    print("KEY TAKEAWAYS:")
    print("  - client.files.create(file, purpose) accepts a (name, BytesIO, mime) tuple")
    print("  - purpose must match intent: 'batch', 'fine-tune', or 'assistants'")
    print("  - client.files.list() returns an iterable of file metadata objects")
    print("  - client.files.retrieve(id) fetches a single file's metadata")
    print("  - client.files.delete(id) removes the file; check .deleted == True")
    print("  - Always delete files after use to avoid unnecessary storage costs")
    print(sep)
