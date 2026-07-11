"""Structured logging utilities for OpenAI API calls."""

from __future__ import annotations

import json
import time
import uuid
from typing import Any


def structured_log(event: str, **kwargs: Any) -> None:
    """Print a JSON-structured log line with timestamp and optional request_id."""
    record: dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "event": event,
        **kwargs,
    }
    print(json.dumps(record))


def new_request_id() -> str:
    """Generate a short unique request ID for tracing."""
    return str(uuid.uuid4())[:8]


class APICallLogger:
    """Context manager that logs API call latency and token usage."""

    def __init__(self, operation: str, request_id: str | None = None) -> None:
        self.operation = operation
        self.request_id = request_id or new_request_id()
        self._start: float = 0.0

    def __enter__(self) -> APICallLogger:
        self._start = time.monotonic()
        structured_log("api_call_start", operation=self.operation, request_id=self.request_id)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        latency_ms = round((time.monotonic() - self._start) * 1000, 1)
        if exc_type is None:
            structured_log(
                "api_call_complete",
                operation=self.operation,
                request_id=self.request_id,
                latency_ms=latency_ms,
            )
        else:
            structured_log(
                "api_call_error",
                operation=self.operation,
                request_id=self.request_id,
                latency_ms=latency_ms,
                error=str(exc_val),
            )
