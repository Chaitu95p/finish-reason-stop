"""Tests for shared.logging — new_request_id, structured_log."""

from __future__ import annotations

from shared.logging import new_request_id, structured_log


def test_new_request_id_is_nonempty() -> None:
    rid = new_request_id()
    assert isinstance(rid, str)
    assert len(rid) > 0


def test_new_request_id_unique() -> None:
    ids = {new_request_id() for _ in range(10)}
    assert len(ids) == 10  # all unique


def test_structured_log_does_not_raise() -> None:
    structured_log("test_event", key="value", count=1)


def test_structured_log_with_various_types() -> None:
    structured_log(
        "test_event",
        string_field="hello",
        int_field=42,
        float_field=3.14,
        bool_field=True,
        none_field=None,
    )
