"""Tests for shared.mock — MockClient and AsyncMockClient."""

from __future__ import annotations

import pytest
from shared.mock import (
    MockClient,
    get_async_client,
    get_client,
    is_mock,
)


def test_is_mock_without_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert is_mock() is True


def test_get_client_returns_mock(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    client = get_client()
    assert isinstance(client, MockClient)


def test_chat_completion_content_and_finish_reason() -> None:
    client = get_client()
    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Hello"}],
    )
    assert resp.choices[0].message.content is not None
    assert resp.choices[0].finish_reason == "stop"


def test_chat_completion_usage() -> None:
    client = get_client()
    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Hello"}],
    )
    assert resp.usage.prompt_tokens > 0
    assert resp.usage.completion_tokens > 0
    assert hasattr(resp.usage, "prompt_tokens_details")
    assert hasattr(resp.usage, "completion_tokens_details")


def test_sync_streaming_context_manager() -> None:
    client = get_client()
    stream = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Hello"}],
        stream=True,
    )
    chunks = []
    with stream:
        for chunk in stream:
            chunks.append(chunk)
    assert len(chunks) > 0
    first = chunks[0]
    assert first.choices[0].delta.content is not None


async def test_async_streaming_context_manager() -> None:
    aclient = await get_async_client()
    stream = await aclient.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Hello"}],
        stream=True,
    )
    chunks = []
    async with stream:
        async for chunk in stream:
            chunks.append(chunk)
    assert len(chunks) > 0
    assert chunks[0].choices[0].delta.content is not None


def test_embeddings_returns_float_list() -> None:
    client = get_client()
    resp = client.embeddings.create(
        model="text-embedding-3-small",
        input="test text",
    )
    emb = resp.data[0].embedding
    assert isinstance(emb, list)
    assert len(emb) > 0
    assert all(isinstance(v, float) for v in emb)


def test_responses_create_status_and_output_text() -> None:
    client = get_client()
    resp = client.responses.create(model="gpt-4o", input="Hello")
    assert resp.status == "completed"
    assert len(resp.output_text) > 0


def test_models_list_has_ids() -> None:
    client = get_client()
    models = client.models.list()
    assert len(models) >= 1
    for m in models:
        assert hasattr(m, "id")
        assert len(m.id) > 0


def test_models_retrieve() -> None:
    client = get_client()
    m = client.models.retrieve("gpt-4o")
    assert m.id == "gpt-4o"


def test_moderations_create_not_flagged() -> None:
    client = get_client()
    result = client.moderations.create(input="Hello, how are you?")
    assert result.results[0].flagged is False
    assert hasattr(result.results[0].categories, "harassment")


def test_usage_prompt_tokens_details() -> None:
    client = get_client()
    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Hello"}],
    )
    assert resp.usage.prompt_tokens_details.cached_tokens >= 0


def test_usage_completion_tokens_details_reasoning() -> None:
    client = get_client()
    resp = client.chat.completions.create(
        model="o3",
        messages=[{"role": "user", "content": "Think about this problem."}],
    )
    assert resp.usage.completion_tokens_details.reasoning_tokens == 150


def test_responses_streaming() -> None:
    client = get_client()
    events = []
    with client.responses.create(model="gpt-4o", input="Hello", stream=True) as stream:
        for event in stream:
            events.append(event)
    assert len(events) > 0
    types = {e.type for e in events}
    assert "response.created" in types
    assert "response.completed" in types


async def test_async_responses_streaming() -> None:
    aclient = await get_async_client()
    events = []
    stream = await aclient.responses.create(model="gpt-4o", input="Hello", stream=True)
    async with stream:
        async for event in stream:
            events.append(event)
    assert len(events) > 0
    assert events[0].type == "response.created"


def test_mock_batch_error_file_id() -> None:
    client = get_client()
    batch = client.batches.retrieve_failed("batch_test001")
    assert batch.status == "failed"
    assert batch.error_file_id == "file-mock-error-001"


def test_realtime_session_create() -> None:
    client = get_client()
    session = client.beta.realtime.sessions.create(model="gpt-4o-realtime-preview")
    assert session.model == "gpt-4o-realtime-preview"
    assert session.status == "created"


async def test_realtime_connection_events() -> None:
    client = get_client()
    events = []
    async with client.beta.realtime.connect() as conn:
        async for evt in conn:
            events.append(evt)
    assert len(events) >= 2
    types = [e.type for e in events]
    assert "session.created" in types
