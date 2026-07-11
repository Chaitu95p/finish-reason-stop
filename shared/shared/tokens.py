"""Token counting and cost estimation utilities."""

from __future__ import annotations

_COST_PER_TOKEN: dict[str, tuple[float, float]] = {
    "gpt-4o": (2.50 / 1_000_000, 10.00 / 1_000_000),
    "gpt-4o-mini": (0.15 / 1_000_000, 0.60 / 1_000_000),
    "gpt-4-turbo": (10.00 / 1_000_000, 30.00 / 1_000_000),
    "gpt-3.5-turbo": (0.50 / 1_000_000, 1.50 / 1_000_000),
    "text-embedding-3-small": (0.02 / 1_000_000, 0.0),
    "text-embedding-3-large": (0.13 / 1_000_000, 0.0),
}


def count_tokens(text: str) -> int:
    """Rough estimate: ~4 characters per token (no tiktoken dependency required)."""
    return max(1, len(text) // 4)


def estimate_cost(
    prompt_tokens: int,
    completion_tokens: int,
    model: str = "gpt-4o",
) -> float:
    """Estimate USD cost from token counts. Returns 0.0 for unknown models."""
    in_rate, out_rate = _COST_PER_TOKEN.get(model, (0.0, 0.0))
    return prompt_tokens * in_rate + completion_tokens * out_rate


def format_cost(cost_usd: float) -> str:
    """Format a cost as a readable string."""
    if cost_usd < 0.001:
        return f"${cost_usd * 100:.4f}¢"
    return f"${cost_usd:.6f}"
