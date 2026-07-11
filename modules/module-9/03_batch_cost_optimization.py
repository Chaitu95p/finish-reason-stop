"""Domain 9 - Task 9.3: Batch Cost Optimization

CONCEPTS:
  1. 50% discount — batch API is half the price of synchronous
  2. Up to 24h latency — the cost of the discount
  3. When to batch — offline processing, large datasets, non-urgent tasks
  4. When NOT to batch — real-time responses, latency-sensitive workloads

Mnemonic: DUAL — Discount_50pct, Unlimited_volume, Async_wait, Latency_tradeoff

Run:
  uv run python 03_batch_cost_optimization.py
"""

NL = chr(10)
MODEL = "gpt-4o-mini"

from shared.mock import is_mock
from shared.tokens import format_cost

SYNC_COST_PER_1M = {"gpt-4o": (2.50, 10.00), "gpt-4o-mini": (0.15, 0.60)}
BATCH_DISCOUNT = 0.50


def calculate_cost(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    batch: bool = False,
) -> float:
    """Calculate API cost; apply 50% discount for batch."""
    in_rate, out_rate = SYNC_COST_PER_1M.get(model, (2.50, 10.00))
    in_rate /= 1_000_000
    out_rate /= 1_000_000
    if batch:
        in_rate *= BATCH_DISCOUNT
        out_rate *= BATCH_DISCOUNT
    return prompt_tokens * in_rate + completion_tokens * out_rate


def demo_cost_comparison() -> None:
    """DEMO 1: Compare sync vs batch costs for a large workload."""
    prompt_tokens = 500
    completion_tokens = 100
    num_requests = 10_000

    for model in ["gpt-4o", "gpt-4o-mini"]:
        sync_per_req = calculate_cost(model, prompt_tokens, completion_tokens, batch=False)
        batch_per_req = calculate_cost(model, prompt_tokens, completion_tokens, batch=True)
        sync_total = sync_per_req * num_requests
        batch_total = batch_per_req * num_requests
        savings = sync_total - batch_total

        print(f"  Model: {model}")
        print(f"    Sync  total ({num_requests:,} requests): {format_cost(sync_total)}")
        print(f"    Batch total ({num_requests:,} requests): {format_cost(batch_total)}")
        print(f"    Savings: {format_cost(savings)} (50%)")
        print()


def demo_decision_matrix() -> None:
    """DEMO 2: When to use batch vs synchronous."""
    print("  Use Batch API when:")
    print("    ✓ Processing large datasets (1,000+ items)")
    print("    ✓ Generating embeddings for a corpus")
    print("    ✓ Overnight batch classification/extraction")
    print("    ✓ Non-user-facing pipeline steps")
    print("    ✓ Cost is more important than latency")
    print(f"{NL}  Use Synchronous API when:")
    print("    ✗ User is waiting for response")
    print("    ✗ Latency matters (chatbots, agents)")
    print("    ✗ Results needed immediately for next step")
    print("    ✗ Fewer than ~100 requests (overhead not worth it)")


def demo_break_even() -> None:
    """DEMO 3: Break-even analysis — when batch overhead pays off."""
    print("  Break-even considerations:")
    print("    Setup overhead: ~1-2 API calls (upload file + create batch)")
    print("    Minimum effective batch size: ~50-100 requests")
    print("    Max requests per batch: 50,000")
    print(f"{NL}  Cost savings by volume:")
    for n in [100, 1_000, 10_000, 100_000]:
        sync = calculate_cost("gpt-4o-mini", 200, 50, batch=False) * n
        batch = calculate_cost("gpt-4o-mini", 200, 50, batch=True) * n
        print(f"    {n:>7,} requests: save {format_cost(sync - batch)}")


if __name__ == "__main__":
    sep = "=" * 60
    mode = "MOCK" if is_mock() else "LIVE"
    print(f"{NL}{sep}{NL}Domain 9 - Task 9.3: Batch Cost Optimization [{mode}]{NL}{sep}")

    print(f"{NL}--- DEMO 1: Cost Comparison ---")
    demo_cost_comparison()

    print(f"{NL}--- DEMO 2: Decision Matrix ---")
    demo_decision_matrix()

    print(f"{NL}--- DEMO 3: Break-Even Analysis ---")
    demo_break_even()

    print(f"{NL}KEY TAKEAWAYS:")
    print("  1. Batch API = 50% discount; latency = up to 24h")
    print("  2. Use batch for offline workloads with 50+ requests")
    print("  3. gpt-4o-mini + batch = 70% cheaper than gpt-4o synchronous")
    print("  4. The setup overhead is ~2 API calls — amortized over large batches")
