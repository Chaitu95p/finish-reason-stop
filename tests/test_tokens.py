"""Tests for shared.tokens — count_tokens, estimate_cost, format_cost."""

from __future__ import annotations

from shared.tokens import count_tokens, estimate_cost, format_cost


def test_count_tokens_positive_for_nonempty_input() -> None:
    count = count_tokens("Hello, world!")
    assert isinstance(count, int)
    assert count > 0


def test_count_tokens_more_for_longer_text() -> None:
    short = count_tokens("Hi")
    long = count_tokens("Hello " * 100)
    assert long > short


def test_estimate_cost_non_negative() -> None:
    cost = estimate_cost(100, 50, "gpt-4o")
    assert isinstance(cost, float)
    assert cost >= 0.0


def test_estimate_cost_zero_tokens() -> None:
    cost = estimate_cost(0, 0, "gpt-4o")
    assert cost == 0.0


def test_format_cost_returns_string() -> None:
    result = format_cost(0.001234)
    assert isinstance(result, str)
    assert len(result) > 0


def test_format_cost_includes_value() -> None:
    result = format_cost(1.5)
    assert "1.5" in result or "1,5" in result or "$" in result
