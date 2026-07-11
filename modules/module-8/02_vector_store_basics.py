"""Domain 8 - Task 8.2: Vector Store Basics

CONCEPTS:
  1. client.vector_stores.create() — create a vector store
  2. client.vector_stores.files.create() — add file to store
  3. Status polling — wait for "completed"
  4. client.vector_stores.delete() — cleanup

Mnemonic: CVSP — Create, Vector_store, Status, Poll

Run:
  uv run python 02_vector_store_basics.py
"""

from __future__ import annotations

import io
import time
from dataclasses import dataclass

from shared.mock import get_client, is_mock

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

NL = chr(10)
MODEL = "gpt-4o"
MAX_POLL_ITERATIONS = 10
POLL_INTERVAL_SECONDS = 1.0


# ---------------------------------------------------------------------------
# Domain objects
# ---------------------------------------------------------------------------


@dataclass
class VectorStoreRecord:
    """Represents a vector store."""

    store_id: str
    name: str
    status: str
    created_at: int
    file_count_total: int = 0
    file_count_completed: int = 0
    file_count_in_progress: int = 0
    file_count_failed: int = 0


@dataclass
class VectorStoreFileRecord:
    """Represents a file attached to a vector store."""

    vsf_id: str
    vector_store_id: str
    status: str


@dataclass
class UploadResult:
    """Tracks the outcome of a file upload + vector store attachment."""

    file_id: str
    filename: str
    vector_store_id: str
    final_status: str
    poll_iterations: int


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def create_vector_store(client: object, name: str) -> VectorStoreRecord:
    """Create a new vector store and return its record."""
    vs = client.vector_stores.create(name=name)  # type: ignore[union-attr]
    fc = vs.file_counts
    return VectorStoreRecord(
        store_id=vs.id,
        name=vs.name,
        status=vs.status,
        created_at=vs.created_at,
        file_count_total=fc.total,
        file_count_completed=fc.completed,
        file_count_in_progress=fc.in_progress,
        file_count_failed=fc.failed,
    )


def upload_file_to_openai(client: object, content: bytes, filename: str) -> str:
    """Upload a raw file to the Files API and return the file_id."""
    file_tuple = (filename, io.BytesIO(content), "text/plain")
    result = client.files.create(file=file_tuple, purpose="assistants")  # type: ignore[union-attr]
    return str(result.id)


def add_file_to_vector_store(
    client: object, vector_store_id: str, file_id: str
) -> VectorStoreFileRecord:
    """Attach a file to a vector store."""
    vsf = client.vector_stores.files.create(  # type: ignore[union-attr]
        vector_store_id=vector_store_id, file_id=file_id
    )
    return VectorStoreFileRecord(
        vsf_id=vsf.id,
        vector_store_id=vsf.vector_store_id,
        status=vsf.status,
    )


def poll_vector_store_status(
    client: object,
    vector_store_id: str,
    max_iterations: int = MAX_POLL_ITERATIONS,
    interval: float = POLL_INTERVAL_SECONDS,
) -> tuple[str, int]:
    """Poll vector store status until completed/failed or max iterations.

    Returns (final_status, iterations_used).
    """
    for i in range(1, max_iterations + 1):
        vs = client.vector_stores.retrieve(vector_store_id)  # type: ignore[union-attr]
        status = vs.status
        if status in ("completed", "failed", "expired"):
            return status, i
        if not is_mock():
            time.sleep(interval)
    return "unknown", max_iterations


def retrieve_vector_store(client: object, vector_store_id: str) -> VectorStoreRecord:
    """Retrieve current state of a vector store."""
    vs = client.vector_stores.retrieve(vector_store_id)  # type: ignore[union-attr]
    fc = vs.file_counts
    return VectorStoreRecord(
        store_id=vs.id,
        name=vs.name,
        status=vs.status,
        created_at=vs.created_at,
        file_count_total=fc.total,
        file_count_completed=fc.completed,
        file_count_in_progress=fc.in_progress,
        file_count_failed=fc.failed,
    )


def delete_vector_store(client: object, vector_store_id: str) -> bool:
    """Delete a vector store. Returns True on success."""
    result = client.vector_stores.delete(vector_store_id)  # type: ignore[union-attr]
    return bool(result.deleted)


def print_vector_store(record: VectorStoreRecord, label: str | None = None) -> None:
    """Print a VectorStoreRecord."""
    if label:
        print(f"  [{label}]")
    print(f"  ID:          {record.store_id}")
    print(f"  Name:        {record.name}")
    print(f"  Status:      {record.status}")
    print(f"  Created at:  {record.created_at}")
    print(f"  Files total: {record.file_count_total}")
    print(f"    completed:   {record.file_count_completed}")
    print(f"    in_progress: {record.file_count_in_progress}")
    print(f"    failed:      {record.file_count_failed}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    client = get_client()
    sep = "=" * 60

    print(sep)
    print("Domain 8 - Task 8.2: Vector Store Basics")
    print(f"Mode: {'MOCK' if is_mock() else 'LIVE'}")
    print(sep)

    # DEMO 1: Create a vector store
    print(NL + "DEMO 1: Create a vector store")
    print("-" * 40)
    store = create_vector_store(client, name="Product Knowledge Base")
    print("  Created vector store:")
    print_vector_store(store, "new store")

    # DEMO 2: Upload files to OpenAI Files API
    print(NL + "DEMO 2: Upload source documents")
    print("-" * 40)
    docs = {
        "install_guide.txt": (
            "Installation Guide" + NL
            + "1. Download the installer" + NL
            + "2. Run ./install.sh" + NL
            + "3. Follow on-screen prompts"
        ),
        "troubleshooting.txt": (
            "Troubleshooting Guide" + NL
            + "- Error 404: Check server URL" + NL
            + "- Error 500: Restart the service" + NL
            + "- Slow response: Check network connectivity"
        ),
    }
    uploaded_file_ids: list[str] = []
    for filename, content in docs.items():
        fid = upload_file_to_openai(client, content.encode("utf-8"), filename)
        uploaded_file_ids.append(fid)
        print(f"  Uploaded: {filename} → file_id={fid}")

    # DEMO 3: Add files to the vector store
    print(NL + "DEMO 3: Add files to vector store")
    print("-" * 40)
    vsf_records: list[VectorStoreFileRecord] = []
    for fid in uploaded_file_ids:
        vsf = add_file_to_vector_store(client, store.store_id, fid)
        vsf_records.append(vsf)
        print(f"  Attached file_id={fid}")
        print(f"    vsf.id={vsf.vsf_id}, status={vsf.status}")

    # DEMO 4: Poll until completed
    print(NL + "DEMO 4: Poll vector store status until completed")
    print("-" * 40)
    print(f"  Polling (max {MAX_POLL_ITERATIONS} iterations)...")
    final_status, iters = poll_vector_store_status(client, store.store_id)
    print(f"  Final status: {final_status} (after {iters} iteration(s))")

    # DEMO 5: Retrieve and inspect final state
    print(NL + "DEMO 5: Retrieve final vector store state")
    print("-" * 40)
    final_store = retrieve_vector_store(client, store.store_id)
    print("  Final vector store state:")
    print_vector_store(final_store, "after polling")

    # DEMO 6: List all vector stores
    print(NL + "DEMO 6: List all vector stores")
    print("-" * 40)
    all_stores = client.vector_stores.list()  # type: ignore[union-attr]
    for idx, vs in enumerate(all_stores.data, 1):
        print(f"  Store {idx}: {vs.id} | {vs.name} | status={vs.status}")

    # DEMO 7: Delete the vector store (cleanup)
    print(NL + "DEMO 7: Delete vector store (cleanup)")
    print("-" * 40)
    ok = delete_vector_store(client, store.store_id)
    print(f"  Deleted {store.store_id}: {'success' if ok else 'FAILED'}")

    print(NL + sep)
    print("KEY TAKEAWAYS:")
    print("  - client.vector_stores.create(name=...) creates an empty store")
    print("  - Files must first be uploaded to Files API before attaching to a store")
    print("  - client.vector_stores.files.create(vector_store_id, file_id) attaches a file")
    print("  - Poll client.vector_stores.retrieve(id).status until 'completed' or 'failed'")
    print("  - file_counts tracks in_progress / completed / failed counts during indexing")
    print("  - Always delete vector stores when done to avoid per-GB-per-day charges")
    print(sep)
