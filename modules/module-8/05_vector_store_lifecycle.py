"""Domain 8 - Task 8.5: Vector Store Lifecycle Management

CONCEPTS:
  1. expires_after — auto-expiry to control cost
  2. File count tracking — in_progress, completed, failed
  3. Cost awareness — charged per GB stored per day
  4. Cleanup strategy — delete when done

Mnemonic: EFCC — Expiry, File_counts, Cost, Cleanup

Run:
  uv run python 05_vector_store_lifecycle.py
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

# Cost constants (as of 2025-07-11 — verify at platform.openai.com/pricing)
VECTOR_STORE_COST_PER_GB_PER_DAY_USD = 0.10
BYTES_PER_GB = 1_073_741_824


# ---------------------------------------------------------------------------
# Domain objects
# ---------------------------------------------------------------------------


@dataclass
class ExpiryPolicy:
    """Describes when a vector store should auto-expire."""

    anchor: str       # "last_active_at" or "created_at"
    days: int         # days from anchor until expiry


@dataclass
class FileCounts:
    """Tracks per-status file counts inside a vector store."""

    in_progress: int = 0
    completed: int = 0
    failed: int = 0
    cancelled: int = 0
    total: int = 0

    @property
    def all_done(self) -> bool:
        """True when no files are still in_progress."""
        return self.in_progress == 0

    @property
    def success_rate(self) -> float:
        """Fraction of completed files out of total (excluding in_progress)."""
        done = self.completed + self.failed + self.cancelled
        return self.completed / done if done > 0 else 0.0


@dataclass
class VectorStoreInfo:
    """Full information about a vector store."""

    store_id: str
    name: str
    status: str
    created_at: int
    file_counts: FileCounts
    expiry_anchor: str | None = None
    expiry_days: int | None = None
    usage_bytes: int = 0


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def create_store_with_expiry(
    client: object,
    name: str,
    expiry: ExpiryPolicy | None = None,
) -> VectorStoreInfo:
    """Create a vector store, optionally with an expires_after policy."""
    kwargs: dict = {"name": name}  # type: ignore[type-arg]
    if expiry:
        kwargs["expires_after"] = {"anchor": expiry.anchor, "days": expiry.days}

    vs = client.vector_stores.create(**kwargs)  # type: ignore[union-attr]
    fc = vs.file_counts
    return VectorStoreInfo(
        store_id=vs.id,
        name=vs.name,
        status=vs.status,
        created_at=vs.created_at,
        file_counts=FileCounts(
            in_progress=fc.in_progress,
            completed=fc.completed,
            failed=fc.failed,
            cancelled=fc.cancelled,
            total=fc.total,
        ),
        expiry_anchor=expiry.anchor if expiry else None,
        expiry_days=expiry.days if expiry else None,
        usage_bytes=getattr(vs, "usage_bytes", 0),
    )


def refresh_store_info(client: object, store_id: str) -> VectorStoreInfo:
    """Retrieve up-to-date metadata for a vector store."""
    vs = client.vector_stores.retrieve(store_id)  # type: ignore[union-attr]
    fc = vs.file_counts
    expires = getattr(vs, "expires_after", None)
    return VectorStoreInfo(
        store_id=vs.id,
        name=vs.name,
        status=vs.status,
        created_at=vs.created_at,
        file_counts=FileCounts(
            in_progress=fc.in_progress,
            completed=fc.completed,
            failed=fc.failed,
            cancelled=fc.cancelled,
            total=fc.total,
        ),
        expiry_anchor=getattr(expires, "anchor", None) if expires else None,
        expiry_days=getattr(expires, "days", None) if expires else None,
        usage_bytes=getattr(vs, "usage_bytes", 0),
    )


def monitor_file_counts(
    client: object,
    store_id: str,
    max_iterations: int = 10,
    interval: float = 1.0,
) -> FileCounts:
    """Poll file counts until all files are out of in_progress state."""
    for i in range(1, max_iterations + 1):
        info = refresh_store_info(client, store_id)
        fc = info.file_counts
        print(
            f"  Poll {i}: total={fc.total}, completed={fc.completed}, "
            f"in_progress={fc.in_progress}, failed={fc.failed}"
        )
        if fc.all_done:
            return fc
        if not is_mock():
            time.sleep(interval)
    return refresh_store_info(client, store_id).file_counts


def estimate_daily_cost_usd(usage_bytes: int) -> float:
    """Estimate daily storage cost in USD given usage_bytes."""
    gb = usage_bytes / BYTES_PER_GB
    return gb * VECTOR_STORE_COST_PER_GB_PER_DAY_USD


def list_all_stores(client: object) -> list[VectorStoreInfo]:
    """Return info for every vector store in the project."""
    result = client.vector_stores.list()  # type: ignore[union-attr]
    stores: list[VectorStoreInfo] = []
    for vs in result.data:
        fc = vs.file_counts
        stores.append(VectorStoreInfo(
            store_id=vs.id,
            name=vs.name,
            status=vs.status,
            created_at=vs.created_at,
            file_counts=FileCounts(
                in_progress=fc.in_progress,
                completed=fc.completed,
                failed=fc.failed,
                cancelled=fc.cancelled,
                total=fc.total,
            ),
            usage_bytes=getattr(vs, "usage_bytes", 0),
        ))
    return stores


def delete_all_vector_stores(client: object) -> int:
    """Delete every vector store in the project. Returns count deleted."""
    stores = list_all_stores(client)
    count = 0
    for store in stores:
        client.vector_stores.delete(store.store_id)  # type: ignore[union-attr]
        print(f"  Deleted: {store.store_id} ({store.name})")
        count += 1
    return count


def print_store_info(info: VectorStoreInfo, label: str | None = None) -> None:
    """Print a VectorStoreInfo in a readable format."""
    if label:
        print(f"  [{label}]")
    print(f"  ID:            {info.store_id}")
    print(f"  Name:          {info.name}")
    print(f"  Status:        {info.status}")
    print(f"  Created at:    {info.created_at}")
    if info.expiry_anchor:
        print(f"  Expires after: {info.expiry_days} days from {info.expiry_anchor}")
    else:
        print("  Expires after: never (no policy set)")
    print(f"  Usage bytes:   {info.usage_bytes:,}")
    daily_cost = estimate_daily_cost_usd(info.usage_bytes)
    print(f"  Est. daily cost: ${daily_cost:.6f}")
    fc = info.file_counts
    print(f"  File counts:   total={fc.total}, completed={fc.completed}, "
          f"in_progress={fc.in_progress}, failed={fc.failed}")
    if fc.total > 0:
        print(f"  Success rate:  {fc.success_rate:.0%}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    client = get_client()
    sep = "=" * 60

    print(sep)
    print("Domain 8 - Task 8.5: Vector Store Lifecycle Management")
    print(f"Mode: {'MOCK' if is_mock() else 'LIVE'}")
    print(sep)

    # DEMO 1: Create a store without expiry (permanent)
    print(NL + "DEMO 1: Create a permanent vector store (no expiry)")
    print("-" * 40)
    permanent_store = create_store_with_expiry(client, name="Permanent Store")
    print_store_info(permanent_store, "permanent")

    # DEMO 2: Create a store with expires_after (last_active_at, 7 days)
    print(NL + "DEMO 2: Create a store with expires_after (7 days from last_active_at)")
    print("-" * 40)
    weekly_policy = ExpiryPolicy(anchor="last_active_at", days=7)
    weekly_store = create_store_with_expiry(
        client, name="Weekly Expiry Store", expiry=weekly_policy
    )
    print_store_info(weekly_store, "weekly expiry")
    print()
    print("  Note: 'last_active_at' resets the clock each time the store is queried.")
    print("  Use 'created_at' for an absolute deadline regardless of usage.")

    # DEMO 3: Create a store with short expiry (1 day) for temporary workloads
    print(NL + "DEMO 3: Short-lived store for a one-off task (1 day expiry)")
    print("-" * 40)
    daily_policy = ExpiryPolicy(anchor="created_at", days=1)
    daily_store = create_store_with_expiry(
        client, name="One-off Task Store", expiry=daily_policy
    )
    print_store_info(daily_store, "1-day expiry")

    # DEMO 4: Add files and monitor file_counts
    print(NL + "DEMO 4: Add files and monitor file_counts")
    print("-" * 40)
    doc_texts = [
        ("doc1.txt", "Chapter 1: Introduction to machine learning."),
        ("doc2.txt", "Chapter 2: Supervised learning algorithms."),
        ("doc3.txt", "Chapter 3: Unsupervised learning methods."),
    ]
    for fname, content in doc_texts:
        file_tuple = (fname, io.BytesIO(content.encode("utf-8")), "text/plain")
        uploaded = client.files.create(file=file_tuple, purpose="assistants")  # type: ignore[union-attr]
        client.vector_stores.files.create(  # type: ignore[union-attr]
            vector_store_id=permanent_store.store_id, file_id=uploaded.id
        )
        print(f"  Attached {fname}")

    print(NL + "  Monitoring file_counts:")
    final_counts = monitor_file_counts(client, permanent_store.store_id)
    print(f"  Final: {final_counts}")

    # DEMO 5: Cost estimation at scale
    print(NL + "DEMO 5: Cost estimation at scale")
    print("-" * 40)
    size_scenarios = [
        ("Small corpus",  50 * 1024 * 1024),          # 50 MB
        ("Medium corpus", 500 * 1024 * 1024),         # 500 MB
        ("Large corpus",  5 * 1024 * 1024 * 1024),   # 5 GB
    ]
    print(f"  {'Scenario':<20} {'Size':>12} {'Daily cost':>15} {'Monthly cost':>15}")
    print("  " + "-" * 64)
    for label, size_bytes in size_scenarios:
        daily = estimate_daily_cost_usd(size_bytes)
        monthly = daily * 30
        size_label = f"{size_bytes / (1024**2):.0f} MB" if size_bytes < BYTES_PER_GB else f"{size_bytes / BYTES_PER_GB:.1f} GB"
        print(f"  {label:<20} {size_label:>12} ${daily:>13.4f} ${monthly:>13.4f}")

    # DEMO 6: List all stores and total estimated cost
    print(NL + "DEMO 6: List all stores with cost estimate")
    print("-" * 40)
    all_stores = list_all_stores(client)
    total_bytes = sum(s.usage_bytes for s in all_stores)
    total_daily = estimate_daily_cost_usd(total_bytes)
    print(f"  Stores found: {len(all_stores)}")
    for s in all_stores:
        daily = estimate_daily_cost_usd(s.usage_bytes)
        print(f"  {s.store_id} | {s.name:<30} | {s.usage_bytes:>10,} bytes | ${daily:.6f}/day")
    print(f"  Total estimated daily cost: ${total_daily:.6f}")

    # DEMO 7: Cleanup — delete all vector stores
    print(NL + "DEMO 7: Cleanup — delete all vector stores")
    print("-" * 40)
    count = delete_all_vector_stores(client)
    print(f"  Deleted {count} vector store(s)")

    # ANTI-PATTERN 1: Leaving vector stores around without expiry
    print(NL + "ANTI-PATTERN 1: Never leave stores without an expiry policy in dev/test")
    print("-" * 40)
    print("  BAD:  client.vector_stores.create(name='temp')")
    print("        No expiry → store lives forever → silent cost accumulation.")
    print("  GOOD: client.vector_stores.create(name='temp',")
    print("          expires_after={'anchor': 'created_at', 'days': 1})")
    print("  Reason: Vector stores are billed at $0.10/GB/day; orphaned stores add up.")

    print(NL + sep)
    print("KEY TAKEAWAYS:")
    print("  - expires_after auto-deletes stores; anchor='last_active_at' extends on use")
    print("  - file_counts.in_progress > 0 means indexing is still running; poll before querying")
    print("  - FileCounts.success_rate reveals whether some files failed to index")
    print("  - Vector stores cost $0.10/GB/day — audit and clean up regularly")
    print("  - Use delete_all_vector_stores() as a teardown function in tests/scripts")
    print("  - Always set expires_after in dev/test environments to avoid surprise bills")
    print(sep)
