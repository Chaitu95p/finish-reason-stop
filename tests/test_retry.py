"""Tests for shared.retry — exponential_backoff decorator."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from shared.retry import exponential_backoff


def test_succeeds_on_first_call() -> None:
    call_count = 0

    @exponential_backoff(max_retries=3, base_delay=0.0, max_delay=0.0)
    def always_succeeds() -> str:
        nonlocal call_count
        call_count += 1
        return "ok"

    result = always_succeeds()
    assert result == "ok"
    assert call_count == 1


def test_retries_on_rate_limit_then_succeeds() -> None:
    import openai

    call_count = 0
    # Build a minimal mock response that openai.RateLimitError accepts
    mock_response = MagicMock()
    mock_response.status_code = 429
    mock_response.headers = {}
    mock_response.request = MagicMock()

    @exponential_backoff(max_retries=3, base_delay=0.0, max_delay=0.0)
    def fails_with_rate_limit() -> str:
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise openai.RateLimitError(
                "rate limited",
                response=mock_response,
                body=None,
            )
        return "success"

    with patch("time.sleep"):
        result = fails_with_rate_limit()

    assert result == "success"
    assert call_count == 3


def test_raises_immediately_for_non_retryable_exception() -> None:
    """Non-OpenAI exceptions propagate immediately without retrying."""
    call_count = 0

    @exponential_backoff(max_retries=3, base_delay=0.0, max_delay=0.0)
    def raises_value_error() -> None:
        nonlocal call_count
        call_count += 1
        raise ValueError("not retryable")

    with pytest.raises(ValueError, match="not retryable"):
        raises_value_error()

    assert call_count == 1  # no retry — non-OpenAI exception propagates immediately


def test_raises_after_max_retries_exhausted() -> None:
    import openai

    call_count = 0
    mock_response = MagicMock()
    mock_response.status_code = 429
    mock_response.headers = {}
    mock_response.request = MagicMock()

    @exponential_backoff(max_retries=2, base_delay=0.0, max_delay=0.0)
    def always_rate_limits() -> None:
        nonlocal call_count
        call_count += 1
        raise openai.RateLimitError(
            "always rate limited",
            response=mock_response,
            body=None,
        )

    with patch("time.sleep"):
        with pytest.raises(openai.RateLimitError):
            always_rate_limits()

    assert call_count == 3  # initial attempt + 2 retries
