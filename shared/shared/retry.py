"""Retry utilities for OpenAI API calls."""

from __future__ import annotations

import functools
import random
import time
from collections.abc import Callable
from typing import Any, TypeVar

F = TypeVar("F", bound=Callable[..., Any])

_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


def exponential_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
) -> Callable[[F], F]:
    """Decorator: retry with exponential backoff on retryable OpenAI errors."""

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            import openai

            retryable = (
                openai.APIConnectionError,
                openai.RateLimitError,
                openai.APITimeoutError,
            )
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except openai.APIStatusError as e:
                    if e.status_code not in _RETRYABLE_STATUS_CODES or attempt == max_retries:
                        raise
                    delay = min(base_delay * (2**attempt) + random.uniform(0, 1), max_delay)
                    time.sleep(delay)
                except retryable:
                    if attempt == max_retries:
                        raise
                    delay = min(base_delay * (2**attempt) + random.uniform(0, 1), max_delay)
                    time.sleep(delay)

        return wrapper  # type: ignore[return-value]

    return decorator
