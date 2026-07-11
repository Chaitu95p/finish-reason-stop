"""Domain 12 - Task 12.2: Cost Tracking

CONCEPTS:
  1. response.usage — prompt_tokens, completion_tokens, total_tokens
  2. Per-request cost — token counts × model rates
  3. Budget enforcement — raise before or after hitting a limit
  4. Cost aggregation — track daily/monthly spend in memory or a store

Mnemonic: UPBA — Usage_field, Per_request, Budget, Aggregate

Run:
  uv run python 02_cost_tracking.py
"""

NL = chr(10)
MODEL = "gpt-4o-mini"

from dataclasses import dataclass, field

from shared.mock import get_client, is_mock
from shared.tokens import estimate_cost, format_cost

PRICING: dict[str, tuple[float, float]] = {
    "gpt-4o":      (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o-realtime-preview": (5.00, 20.00),
}


@dataclass
class CostTracker:
    budget_usd: float
    _total_cost: float = field(default=0.0, init=False)
    _call_count: int = field(default=0, init=False)

    def record(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        """Record a call; return the cost of this call."""
        cost = estimate_cost(prompt_tokens, completion_tokens, model)
        self._total_cost += cost
        self._call_count += 1
        return cost

    def check_budget(self) -> None:
        """Raise RuntimeError if total cost exceeds budget."""
        if self._total_cost > self.budget_usd:
            raise RuntimeError(
                f"Budget exceeded: spent {format_cost(self._total_cost)}"
                f" of {format_cost(self.budget_usd)} budget"
            )

    @property
    def total_cost(self) -> float:
        return self._total_cost

    @property
    def remaining_budget(self) -> float:
        return max(0.0, self.budget_usd - self._total_cost)

    def summary(self) -> str:
        return (
            f"Calls: {self._call_count} | "
            f"Spent: {format_cost(self._total_cost)} | "
            f"Remaining: {format_cost(self.remaining_budget)}"
        )


def demo_usage_field() -> None:
    """DEMO 1: Reading token counts from response.usage."""
    client = get_client()
    response = client.chat.completions.create(  # type: ignore[attr-defined]
        model=MODEL,
        messages=[{"role": "user", "content": "What is 2+2?"}],
    )
    usage = response.usage
    prompt_tokens = getattr(usage, "prompt_tokens", 10)
    completion_tokens = getattr(usage, "completion_tokens", 5)
    total_tokens = getattr(usage, "total_tokens", 15)

    cost = estimate_cost(prompt_tokens, completion_tokens, MODEL)
    print(f"  prompt_tokens:     {prompt_tokens}")
    print(f"  completion_tokens: {completion_tokens}")
    print(f"  total_tokens:      {total_tokens}")
    print(f"  cost:              {format_cost(cost)}")


def demo_budget_tracker() -> None:
    """DEMO 2: CostTracker with budget enforcement."""
    client = get_client()
    tracker = CostTracker(budget_usd=0.01)
    prompts = [
        "Name a color.",
        "Name an animal.",
        "Name a country.",
    ]
    for prompt in prompts:
        response = client.chat.completions.create(  # type: ignore[attr-defined]
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        cost = tracker.record(MODEL, 5, 3)
        try:
            tracker.check_budget()
        except RuntimeError as e:
            print(f"  BUDGET EXCEEDED: {e}")
            break
        print(f"  [{prompt!r}] cost={format_cost(cost)} | {tracker.summary()}")


def demo_pricing_table() -> None:
    """DEMO 3: Model pricing reference."""
    print(f"  {'Model':<30} {'Input/1M':<12} {'Output/1M'}")
    print(f"  {'-'*30} {'-'*12} {'-'*10}")
    for model, (inp, out) in PRICING.items():
        print(f"  {model:<30} ${inp:<11.2f} ${out:.2f}")


def demo_cost_optimization_tips() -> None:
    """DEMO 4: Practical cost reduction strategies."""
    print("  Cost optimization strategies:")
    print("    1. Use gpt-4o-mini for simple tasks — 17× cheaper than gpt-4o")
    print("    2. Set max_tokens — prevent unexpectedly long completions")
    print("    3. Batch API — 50% discount for async workloads")
    print("    4. Prompt caching — repeated prefixes count as cached tokens (reduced rate)")
    print("    5. Trim conversation history — only send last N turns in multi-turn chats")
    print(f"{NL}  Example: 1M daily requests at 500 tokens each:")
    tokens = 1_000_000 * 500
    mini_cost = (tokens / 1_000_000) * 0.15
    o_cost = (tokens / 1_000_000) * 2.50
    print(f"    gpt-4o-mini: {format_cost(mini_cost)}/day")
    print(f"    gpt-4o:      {format_cost(o_cost)}/day ({o_cost / mini_cost:.0f}× more expensive)")


if __name__ == "__main__":
    sep = "=" * 60
    mode = "MOCK" if is_mock() else "LIVE"
    print(f"{NL}{sep}{NL}Domain 12 - Task 12.2: Cost Tracking [{mode}]{NL}{sep}")

    print(f"{NL}--- DEMO 1: Reading Usage Field ---")
    demo_usage_field()

    print(f"{NL}--- DEMO 2: Budget Tracker ---")
    demo_budget_tracker()

    print(f"{NL}--- DEMO 3: Pricing Table ---")
    demo_pricing_table()

    print(f"{NL}--- DEMO 4: Cost Optimization ---")
    demo_cost_optimization_tips()

    print(f"{NL}KEY TAKEAWAYS:")
    print("  1. response.usage has prompt_tokens, completion_tokens, total_tokens")
    print("  2. Track cumulative cost per session/user/day; alert before exceeding budget")
    print("  3. gpt-4o-mini is 17× cheaper than gpt-4o for input tokens")
    print("  4. Batch API gives 50% discount — use for offline/async workloads")
