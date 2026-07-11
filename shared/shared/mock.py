"""Mock client for OpenAI SDK — enables all scripts to run without an API key.

get_client() returns real OpenAI() when OPENAI_API_KEY is set, otherwise MockClient.
get_async_client() returns real AsyncOpenAI() or AsyncMockClient.
"""

from __future__ import annotations

import asyncio
import json
import os
import types
from collections.abc import AsyncGenerator, Iterator
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------


def is_mock() -> bool:
    """True when OPENAI_API_KEY is absent or empty."""
    return not bool(os.environ.get("OPENAI_API_KEY", "").strip())


# ---------------------------------------------------------------------------
# Module-level state for prompt cache simulation
# ---------------------------------------------------------------------------

_prompt_cache: dict[str, int] = {}


# ---------------------------------------------------------------------------
# Response dataclasses — duck-typed equivalents of openai response objects
# ---------------------------------------------------------------------------


@dataclass
class MockPromptTokensDetails:
    cached_tokens: int = 0


@dataclass
class MockCompletionTokensDetails:
    reasoning_tokens: int = 0


@dataclass
class MockUsage:
    prompt_tokens: int = 20
    completion_tokens: int = 30
    total_tokens: int = 50
    prompt_tokens_details: MockPromptTokensDetails = field(
        default_factory=MockPromptTokensDetails
    )
    completion_tokens_details: MockCompletionTokensDetails = field(
        default_factory=MockCompletionTokensDetails
    )


@dataclass
class MockFunction:
    name: str = "mock_function"
    arguments: str = "{}"


@dataclass
class MockToolCall:
    id: str = "call_mock001"
    type: str = "function"
    function: MockFunction = field(default_factory=MockFunction)


@dataclass
class MockMessage:
    content: str | None = "Mock response content."
    role: str = "assistant"
    tool_calls: list[MockToolCall] = field(default_factory=list)
    refusal: str | None = None


@dataclass
class MockChoice:
    finish_reason: str = "stop"
    index: int = 0
    message: MockMessage = field(default_factory=MockMessage)

    def __post_init__(self) -> None:
        if self.finish_reason == "tool_calls" and not self.message.tool_calls:
            self.message.tool_calls = [MockToolCall()]


@dataclass
class MockChatCompletion:
    choices: list[MockChoice] = field(default_factory=list)
    model: str = "gpt-4o"
    id: str = "chatcmpl-mock001"
    usage: MockUsage = field(default_factory=MockUsage)
    object: str = "chat.completion"

    def __post_init__(self) -> None:
        if not self.choices:
            self.choices = [MockChoice()]


@dataclass
class MockResponseOutput:
    text: str = "Mock output text."
    type: str = "output_text"
    id: str = "msg_mock001"

    # For tool_call type outputs
    name: str | None = None
    arguments: str | None = None
    call_id: str | None = None


@dataclass
class MockResponse:
    output: list[MockResponseOutput] = field(default_factory=list)
    output_text: str = "Mock output text."
    status: str = "completed"
    id: str = "resp_mock001"
    model: str = "gpt-4o"
    usage: MockUsage = field(default_factory=MockUsage)

    def __post_init__(self) -> None:
        if not self.output:
            self.output = [MockResponseOutput(text=self.output_text)]


@dataclass
class MockEmbeddingData:
    embedding: list[float] = field(default_factory=lambda: [0.1] * 1536)
    index: int = 0
    object: str = "embedding"


@dataclass
class MockCreateEmbeddingResponse:
    data: list[MockEmbeddingData] = field(default_factory=list)
    model: str = "text-embedding-3-small"
    usage: MockUsage = field(default_factory=MockUsage)
    object: str = "list"

    def __post_init__(self) -> None:
        if not self.data:
            self.data = [MockEmbeddingData()]


@dataclass
class MockStreamDelta:
    content: str | None = None
    role: str | None = None
    tool_calls: list[Any] | None = None


@dataclass
class MockStreamChoice:
    delta: MockStreamDelta = field(default_factory=MockStreamDelta)
    finish_reason: str | None = None
    index: int = 0


@dataclass
class MockStreamChunk:
    choices: list[MockStreamChoice] = field(default_factory=list)
    id: str = "chatcmpl-mock-stream"
    model: str = "gpt-4o"
    object: str = "chat.completion.chunk"

    def __post_init__(self) -> None:
        if not self.choices:
            self.choices = [MockStreamChoice()]


@dataclass
class MockImageData:
    url: str = "https://mock-image.example.com/img.png"
    b64_json: str | None = None
    revised_prompt: str | None = None


@dataclass
class MockImageResponse:
    data: list[MockImageData] = field(default_factory=list)
    created: int = 1700000000

    def __post_init__(self) -> None:
        if not self.data:
            self.data = [MockImageData()]


@dataclass
class MockFile:
    id: str = "file-mock001"
    filename: str = "mock_file.jsonl"
    purpose: str = "batch"
    status: str = "processed"
    bytes: int = 1024
    created_at: int = 1700000000
    object: str = "file"


@dataclass
class MockBatch:
    id: str = "batch_mock001"
    status: str = "completed"
    endpoint: str = "/v1/chat/completions"
    completion_window: str = "24h"
    input_file_id: str = "file-mock001"
    output_file_id: str = "file-mock002"
    error_file_id: str | None = None
    created_at: int = 1700000000
    object: str = "batch"


@dataclass
class MockVectorStore:
    id: str = "vs_mock001"
    name: str = "Mock Vector Store"
    status: str = "completed"
    file_counts: Any = field(default_factory=lambda: types.SimpleNamespace(
        in_progress=0, completed=1, failed=0, cancelled=0, total=1
    ))
    created_at: int = 1700000000
    object: str = "vector_store"


@dataclass
class MockVectorStoreFile:
    id: str = "vsf_mock001"
    vector_store_id: str = "vs_mock001"
    status: str = "completed"
    object: str = "vector_store.file"


@dataclass
class MockAssistant:
    id: str = "asst_mock001"
    model: str = "gpt-4o"
    name: str = "Mock Assistant"
    instructions: str | None = "You are a helpful assistant."
    tools: list[Any] = field(default_factory=list)
    object: str = "assistant"
    created_at: int = 1700000000


@dataclass
class MockThread:
    id: str = "thread_mock001"
    object: str = "thread"
    created_at: int = 1700000000


@dataclass
class MockThreadMessage:
    id: str = "msg_mock001"
    thread_id: str = "thread_mock001"
    role: str = "user"
    content: list[Any] = field(default_factory=list)
    object: str = "thread.message"
    created_at: int = 1700000000


@dataclass
class MockRun:
    id: str = "run_mock001"
    thread_id: str = "thread_mock001"
    assistant_id: str = "asst_mock001"
    status: str = "completed"
    model: str = "gpt-4o"
    object: str = "thread.run"
    created_at: int = 1700000000


@dataclass
class MockFineTuningJob:
    id: str = "ftjob_mock001"
    model: str = "gpt-4o-mini"
    status: str = "succeeded"
    fine_tuned_model: str = "ft:gpt-4o-mini:org:suffix:id"
    training_file: str = "file-mock001"
    object: str = "fine_tuning.job"
    created_at: int = 1700000000


@dataclass
class MockModel:
    id: str = "gpt-4o"
    created: int = 1699000000
    owned_by: str = "openai"
    object: str = "model"


@dataclass
class MockModerationCategories:
    harassment: bool = False
    harassment_threatening: bool = False
    hate: bool = False
    hate_threatening: bool = False
    self_harm: bool = False
    self_harm_instructions: bool = False
    self_harm_intent: bool = False
    sexual: bool = False
    sexual_minors: bool = False
    violence: bool = False
    violence_graphic: bool = False


@dataclass
class MockModerationResult:
    flagged: bool = False
    categories: MockModerationCategories = field(
        default_factory=MockModerationCategories
    )


@dataclass
class MockModerationResponse:
    results: list[MockModerationResult] = field(
        default_factory=lambda: [MockModerationResult()]
    )


@dataclass
class MockRealtimeSession:
    id: str = "sess_mock_001"
    model: str = "gpt-4o-realtime-preview"
    modalities: list[str] = field(default_factory=lambda: ["text", "audio"])
    status: str = "created"


@dataclass
class MockRealtimeEvent:
    type: str = "session.created"
    event_id: str = "evt_mock_001"


@dataclass
class MockResponseEvent:
    type: str = "response.created"
    event_id: str = "evt_mock_001"
    delta: str = ""
    output_text: str = ""


# ---------------------------------------------------------------------------
# Namespace classes — mirror openai.OpenAI attribute hierarchy
# ---------------------------------------------------------------------------


def _stream_chunks(text: str = "Mock streaming response.") -> Iterator[MockStreamChunk]:
    """Yield mock stream chunks word by word."""
    words = text.split()
    for i, word in enumerate(words):
        delta = MockStreamDelta(content=word + (" " if i < len(words) - 1 else ""))
        chunk = MockStreamChunk(choices=[MockStreamChoice(delta=delta)])
        yield chunk
    final_delta = MockStreamDelta(content=None)
    yield MockStreamChunk(choices=[MockStreamChoice(delta=final_delta, finish_reason="stop")])


class _MockStreamContextManager:
    """Context manager returned when stream=True; works as both sync and async."""

    def __init__(self, text: str = "Mock streaming response.") -> None:
        self._text = text
        self._chunks = list(_stream_chunks(text))

    def __enter__(self) -> _MockStreamContextManager:
        return self

    def __exit__(self, *_: Any) -> None:
        pass

    def __iter__(self) -> Iterator[MockStreamChunk]:
        yield from self._chunks

    async def __aenter__(self) -> _MockStreamContextManager:
        return self

    async def __aexit__(self, *_: Any) -> None:
        pass

    def __aiter__(self) -> AsyncGenerator[MockStreamChunk, None]:
        return self._async_gen()

    async def _async_gen(self) -> AsyncGenerator[MockStreamChunk, None]:
        for chunk in self._chunks:
            yield chunk

    def get_final_completion(self) -> MockChatCompletion:
        return MockChatCompletion()


class _MockResponsesStreamContextManager:
    """Context manager for Responses API streaming; yields MockResponseEvent objects."""

    _EVENTS = [
        MockResponseEvent(type="response.created", event_id="evt_001"),
        MockResponseEvent(
            type="response.content_part.delta", event_id="evt_002", delta="Hello"
        ),
        MockResponseEvent(
            type="response.content_part.delta", event_id="evt_003", delta=" world."
        ),
        MockResponseEvent(
            type="response.completed",
            event_id="evt_004",
            output_text="Hello world.",
        ),
    ]

    def __enter__(self) -> _MockResponsesStreamContextManager:
        return self

    def __exit__(self, *_: Any) -> None:
        pass

    def __iter__(self) -> Iterator[MockResponseEvent]:
        yield from self._EVENTS

    async def __aenter__(self) -> _MockResponsesStreamContextManager:
        return self

    async def __aexit__(self, *_: Any) -> None:
        pass

    def __aiter__(self) -> AsyncGenerator[MockResponseEvent, None]:
        return self._async_gen()

    async def _async_gen(self) -> AsyncGenerator[MockResponseEvent, None]:
        for evt in self._EVENTS:
            yield evt


def _is_reasoning_model(model: str) -> bool:
    return model.startswith(("o1", "o3", "o4"))


def _make_usage(
    model: str = "gpt-4o",
    messages: list[Any] | None = None,
    cached_tokens: int | None = None,
) -> MockUsage:
    """Build MockUsage with reasoning and cache fields set appropriately."""
    reasoning = 150 if _is_reasoning_model(model) else 0
    if cached_tokens is None:
        prompt_key = str(messages)[:200] if messages else ""
        if prompt_key and prompt_key in _prompt_cache:
            cached = 500
        elif prompt_key:
            _prompt_cache[prompt_key] = 1
            cached = 0
        else:
            cached = 0
    else:
        cached = cached_tokens
    return MockUsage(
        prompt_tokens_details=MockPromptTokensDetails(cached_tokens=cached),
        completion_tokens_details=MockCompletionTokensDetails(reasoning_tokens=reasoning),
    )


class _ChatCompletionsNamespace:
    def create(
        self,
        model: str = "gpt-4o",
        messages: list[Any] | None = None,
        stream: bool = False,
        tools: list[Any] | None = None,
        tool_choice: Any = "auto",
        response_format: Any = None,
        **kwargs: Any,
    ) -> MockChatCompletion | _MockStreamContextManager:
        if stream:
            return _MockStreamContextManager()
        usage = _make_usage(model=model, messages=messages)
        if tools and tool_choice == "required":
            tool = tools[0]
            fn_name = tool.get("function", {}).get("name", "mock_function")
            tool_call = MockToolCall(
                function=MockFunction(name=fn_name, arguments='{"mock_arg": "mock_value"}')
            )
            msg = MockMessage(content=None, tool_calls=[tool_call])
            return MockChatCompletion(
                choices=[MockChoice(finish_reason="tool_calls", message=msg)],
                model=model,
                usage=usage,
            )
        # Return content=None for json_object so scripts' `or json.dumps({...})` fallbacks trigger
        if isinstance(response_format, dict) and response_format.get("type") == "json_object":
            msg = MockMessage(content=None)
            return MockChatCompletion(
                choices=[MockChoice(message=msg)], model=model, usage=usage
            )
        return MockChatCompletion(model=model, usage=usage)

    def parse(
        self,
        model: str = "gpt-4o",
        messages: list[Any] | None = None,
        response_format: Any = None,
        **kwargs: Any,
    ) -> Any:
        """Mock for client.beta.chat.completions.parse()."""
        if response_format is not None and hasattr(response_format, "model_fields"):
            try:
                instance = response_format.model_construct(
                    **{k: "mock" for k in response_format.model_fields}
                )
                msg = MockMessage(content=None)
                msg.parsed = instance  # type: ignore[attr-defined]
                return MockChatCompletion(choices=[MockChoice(message=msg)])
            except Exception:
                pass
        return MockChatCompletion()


class _BetaChatCompletionsNamespace:
    def __init__(self) -> None:
        self._inner = _ChatCompletionsNamespace()

    def parse(self, **kwargs: Any) -> Any:
        return self._inner.parse(**kwargs)


class _BetaChatNamespace:
    def __init__(self) -> None:
        self.completions = _BetaChatCompletionsNamespace()


class _ChatNamespace:
    def __init__(self) -> None:
        self.completions = _ChatCompletionsNamespace()


class _ResponsesNamespace:
    def create(
        self,
        model: str = "gpt-4o",
        input: Any = "",
        tools: list[Any] | None = None,
        previous_response_id: str | None = None,
        instructions: str | None = None,
        tool_choice: Any = "auto",
        stream: bool = False,
        **kwargs: Any,
    ) -> MockResponse | _MockResponsesStreamContextManager:
        if stream:
            return _MockResponsesStreamContextManager()
        if tools and tool_choice == "required":
            tool = tools[0]
            fn_name = tool.get("name", "mock_function")
            tool_output = MockResponseOutput(
                type="tool_call",
                name=fn_name,
                arguments='{"mock_arg": "mock_value"}',
                call_id="call_mock001",
            )
            return MockResponse(
                output=[tool_output],
                output_text="",
                status="completed",
            )
        return MockResponse()


class _EmbeddingsNamespace:
    def create(
        self,
        model: str = "text-embedding-3-small",
        input: str | list[str] = "",
        dimensions: int | None = None,
        **kwargs: Any,
    ) -> MockCreateEmbeddingResponse:
        dims = dimensions or 1536
        if isinstance(input, list):
            data = [
                MockEmbeddingData(embedding=[0.1] * dims, index=i)
                for i in range(len(input))
            ]
        else:
            data = [MockEmbeddingData(embedding=[0.1] * dims)]
        return MockCreateEmbeddingResponse(data=data)


class _ImagesNamespace:
    def generate(
        self,
        model: str = "dall-e-3",
        prompt: str = "",
        n: int = 1,
        size: str = "1024x1024",
        **kwargs: Any,
    ) -> MockImageResponse:
        return MockImageResponse()

    def edit(self, **kwargs: Any) -> MockImageResponse:
        return MockImageResponse()

    def create_variation(self, **kwargs: Any) -> MockImageResponse:
        return MockImageResponse()


class _AudioTranscriptionsNamespace:
    def create(self, model: str = "whisper-1", file: Any = None, **kwargs: Any) -> Any:
        return types.SimpleNamespace(text="Mock transcription text.")


class _AudioTranslationsNamespace:
    def create(self, model: str = "whisper-1", file: Any = None, **kwargs: Any) -> Any:
        return types.SimpleNamespace(text="Mock translation text.")


class _AudioSpeechNamespace:
    def create(
        self,
        model: str = "tts-1",
        voice: str = "alloy",
        input: str = "",
        **kwargs: Any,
    ) -> Any:
        return types.SimpleNamespace(content=b"mock_audio_bytes")

    def stream(self, **kwargs: Any) -> Any:
        return self.create(**kwargs)


class _AudioNamespace:
    def __init__(self) -> None:
        self.transcriptions = _AudioTranscriptionsNamespace()
        self.translations = _AudioTranslationsNamespace()
        self.speech = _AudioSpeechNamespace()


class _FilesListResponse:
    def __init__(self) -> None:
        self.data = [MockFile()]

    def __iter__(self) -> Iterator[MockFile]:
        yield from self.data


class _FilesNamespace:
    def create(self, file: Any = None, purpose: str = "batch", **kwargs: Any) -> MockFile:
        return MockFile(purpose=purpose)

    def list(self, **kwargs: Any) -> _FilesListResponse:
        return _FilesListResponse()

    def retrieve(self, file_id: str, **kwargs: Any) -> MockFile:
        return MockFile(id=file_id)

    def delete(self, file_id: str, **kwargs: Any) -> Any:
        return types.SimpleNamespace(id=file_id, deleted=True)

    def content(self, file_id: str, **kwargs: Any) -> Any:
        line = json.dumps({
            "custom_id": "req-001",
            "response": {
                "status_code": 200,
                "body": {
                    "choices": [{"message": {"content": "Mock batch result."}}]
                },
            },
        })
        return types.SimpleNamespace(content=line.encode())


class _BatchesNamespace:
    def create(
        self,
        input_file_id: str,
        endpoint: str = "/v1/chat/completions",
        completion_window: str = "24h",
        **kwargs: Any,
    ) -> MockBatch:
        return MockBatch(input_file_id=input_file_id)

    def retrieve(self, batch_id: str, **kwargs: Any) -> MockBatch:
        return MockBatch(id=batch_id)

    def retrieve_failed(self, batch_id: str, **kwargs: Any) -> MockBatch:
        """Return a failed batch with error_file_id set."""
        return MockBatch(
            id=batch_id,
            status="failed",
            error_file_id="file-mock-error-001",
        )

    def list(self, **kwargs: Any) -> Any:
        return types.SimpleNamespace(data=[MockBatch()])


class _VectorStoreFilesNamespace:
    def create(self, vector_store_id: str, file_id: str, **kwargs: Any) -> MockVectorStoreFile:
        return MockVectorStoreFile(vector_store_id=vector_store_id)

    def list(self, vector_store_id: str, **kwargs: Any) -> Any:
        return types.SimpleNamespace(data=[MockVectorStoreFile(vector_store_id=vector_store_id)])

    def retrieve(self, vector_store_id: str, file_id: str, **kwargs: Any) -> MockVectorStoreFile:
        return MockVectorStoreFile(id=file_id, vector_store_id=vector_store_id)


class _VectorStoresNamespace:
    def __init__(self) -> None:
        self.files = _VectorStoreFilesNamespace()

    def create(self, name: str = "Mock Store", **kwargs: Any) -> MockVectorStore:
        return MockVectorStore(name=name)

    def retrieve(self, vector_store_id: str, **kwargs: Any) -> MockVectorStore:
        return MockVectorStore(id=vector_store_id)

    def delete(self, vector_store_id: str, **kwargs: Any) -> Any:
        return types.SimpleNamespace(id=vector_store_id, deleted=True)

    def list(self, **kwargs: Any) -> Any:
        return types.SimpleNamespace(data=[MockVectorStore()])


class _AssistantsNamespace:
    def create(self, model: str = "gpt-4o", **kwargs: Any) -> MockAssistant:
        return MockAssistant(model=model, **{
            k: v for k, v in kwargs.items()
            if k in ("name", "instructions", "tools")
        })

    def retrieve(self, assistant_id: str, **kwargs: Any) -> MockAssistant:
        return MockAssistant(id=assistant_id)

    def list(self, **kwargs: Any) -> Any:
        return types.SimpleNamespace(data=[MockAssistant()])

    def delete(self, assistant_id: str, **kwargs: Any) -> Any:
        return types.SimpleNamespace(id=assistant_id, deleted=True)


class _ThreadMessagesNamespace:
    def create(
        self,
        thread_id: str,
        role: str = "user",
        content: str = "",
        **kwargs: Any,
    ) -> MockThreadMessage:
        return MockThreadMessage(thread_id=thread_id, role=role)

    def list(self, thread_id: str, **kwargs: Any) -> Any:
        msg = MockThreadMessage(thread_id=thread_id, role="assistant")
        msg.content = [types.SimpleNamespace(
            type="text",
            text=types.SimpleNamespace(value="Mock assistant response."),
        )]
        return types.SimpleNamespace(data=[msg])


class _ThreadRunsNamespace:
    def create(
        self,
        thread_id: str,
        assistant_id: str = "asst_mock001",
        **kwargs: Any,
    ) -> MockRun:
        return MockRun(thread_id=thread_id, assistant_id=assistant_id)

    def retrieve(self, thread_id: str, run_id: str, **kwargs: Any) -> MockRun:
        return MockRun(id=run_id, thread_id=thread_id)

    def stream(self, thread_id: str, assistant_id: str, **kwargs: Any) -> Any:
        return _MockStreamContextManager("Mock run streaming response.")


class _ThreadsNamespace:
    def __init__(self) -> None:
        self.messages = _ThreadMessagesNamespace()
        self.runs = _ThreadRunsNamespace()

    def create(self, **kwargs: Any) -> MockThread:
        return MockThread()

    def retrieve(self, thread_id: str, **kwargs: Any) -> MockThread:
        return MockThread(id=thread_id)

    def delete(self, thread_id: str, **kwargs: Any) -> Any:
        return types.SimpleNamespace(id=thread_id, deleted=True)


class _FineTuningJobsNamespace:
    def create(
        self,
        training_file: str,
        model: str = "gpt-4o-mini",
        **kwargs: Any,
    ) -> MockFineTuningJob:
        return MockFineTuningJob(training_file=training_file, model=model)

    def retrieve(self, fine_tuning_job_id: str, **kwargs: Any) -> MockFineTuningJob:
        return MockFineTuningJob(id=fine_tuning_job_id)

    def list(self, **kwargs: Any) -> Any:
        return types.SimpleNamespace(data=[MockFineTuningJob()])

    def list_events(self, fine_tuning_job_id: str, **kwargs: Any) -> Any:
        events = [
            types.SimpleNamespace(message="Fine-tuning job started.", level="info"),
            types.SimpleNamespace(message="Step 1/10.", level="info"),
            types.SimpleNamespace(message="Fine-tuning job succeeded!", level="info"),
        ]
        return types.SimpleNamespace(data=events)


class _FineTuningNamespace:
    def __init__(self) -> None:
        self.jobs = _FineTuningJobsNamespace()


class _ModelsNamespace:
    _MODEL_IDS = ["gpt-4o", "gpt-4o-mini", "o3", "text-embedding-3-small"]

    def list(self) -> list[MockModel]:
        return [MockModel(id=m) for m in self._MODEL_IDS]

    def retrieve(self, model_id: str) -> MockModel:
        return MockModel(id=model_id)


class _ModerationsNamespace:
    def create(self, input: str = "", **kwargs: Any) -> MockModerationResponse:
        return MockModerationResponse()


class _RealtimeSessionsNamespace:
    def create(
        self,
        model: str = "gpt-4o-realtime-preview",
        **kwargs: Any,
    ) -> MockRealtimeSession:
        return MockRealtimeSession(model=model)


class MockRealtimeConnection:
    """Async context manager yielding mock realtime events."""

    _EVENTS = [
        MockRealtimeEvent(type="session.created", event_id="evt_001"),
        MockRealtimeEvent(type="response.text.delta", event_id="evt_002"),
        MockRealtimeEvent(type="response.text.done", event_id="evt_003"),
        MockRealtimeEvent(type="session.closed", event_id="evt_004"),
    ]

    async def __aenter__(self) -> MockRealtimeConnection:
        return self

    async def __aexit__(self, *_: Any) -> None:
        pass

    def __aiter__(self) -> AsyncGenerator[MockRealtimeEvent, None]:
        return self._async_gen()

    async def _async_gen(self) -> AsyncGenerator[MockRealtimeEvent, None]:
        for evt in self._EVENTS:
            yield evt

    def send(self, event: Any) -> None:
        pass  # no-op in mock


class _RealtimeNamespace:
    def __init__(self) -> None:
        self.sessions = _RealtimeSessionsNamespace()

    def connect(self, **kwargs: Any) -> MockRealtimeConnection:
        return MockRealtimeConnection()


class _BetaNamespace:
    def __init__(self) -> None:
        self.chat = _BetaChatNamespace()
        self.assistants = _AssistantsNamespace()
        self.threads = _ThreadsNamespace()
        self.realtime = _RealtimeNamespace()


# ---------------------------------------------------------------------------
# MockClient — synchronous
# ---------------------------------------------------------------------------


class MockClient:
    """Synchronous mock drop-in for openai.OpenAI. All namespaces present."""

    def __init__(self) -> None:
        self.responses = _ResponsesNamespace()
        self.chat = _ChatNamespace()
        self.embeddings = _EmbeddingsNamespace()
        self.images = _ImagesNamespace()
        self.audio = _AudioNamespace()
        self.files = _FilesNamespace()
        self.batches = _BatchesNamespace()
        self.vector_stores = _VectorStoresNamespace()
        self.fine_tuning = _FineTuningNamespace()
        self.models = _ModelsNamespace()
        self.moderations = _ModerationsNamespace()
        self.beta = _BetaNamespace()


# ---------------------------------------------------------------------------
# AsyncMockClient — asynchronous (mirrors MockClient exactly)
# ---------------------------------------------------------------------------


class _AsyncChatCompletionsNamespace:
    async def create(self, **kwargs: Any) -> MockChatCompletion | _MockStreamContextManager:
        await asyncio.sleep(0)
        return _ChatCompletionsNamespace().create(**kwargs)

    async def parse(self, **kwargs: Any) -> Any:
        await asyncio.sleep(0)
        return _ChatCompletionsNamespace().parse(**kwargs)


class _AsyncBetaChatCompletionsNamespace:
    async def parse(self, **kwargs: Any) -> Any:
        await asyncio.sleep(0)
        return _ChatCompletionsNamespace().parse(**kwargs)


class _AsyncBetaChatNamespace:
    def __init__(self) -> None:
        self.completions = _AsyncBetaChatCompletionsNamespace()


class _AsyncChatNamespace:
    def __init__(self) -> None:
        self.completions = _AsyncChatCompletionsNamespace()


class _AsyncResponsesNamespace:
    async def create(
        self, **kwargs: Any
    ) -> MockResponse | _MockResponsesStreamContextManager:
        await asyncio.sleep(0)
        return _ResponsesNamespace().create(**kwargs)


class _AsyncEmbeddingsNamespace:
    async def create(self, **kwargs: Any) -> MockCreateEmbeddingResponse:
        await asyncio.sleep(0)
        return _EmbeddingsNamespace().create(**kwargs)


class _AsyncBetaNamespace:
    def __init__(self) -> None:
        self.chat = _AsyncBetaChatNamespace()
        self.assistants = _AssistantsNamespace()
        self.threads = _ThreadsNamespace()
        self.realtime = _RealtimeNamespace()


class AsyncMockClient:
    """Async mock drop-in for openai.AsyncOpenAI. All namespaces present."""

    def __init__(self) -> None:
        self.responses = _AsyncResponsesNamespace()
        self.chat = _AsyncChatNamespace()
        self.embeddings = _AsyncEmbeddingsNamespace()
        self.images = _ImagesNamespace()
        self.audio = _AudioNamespace()
        self.files = _FilesNamespace()
        self.batches = _BatchesNamespace()
        self.vector_stores = _VectorStoresNamespace()
        self.fine_tuning = _FineTuningNamespace()
        self.models = _ModelsNamespace()
        self.moderations = _ModerationsNamespace()
        self.beta = _AsyncBetaNamespace()


# ---------------------------------------------------------------------------
# Public factory functions
# ---------------------------------------------------------------------------


def get_client() -> Any:
    """Return MockClient if no API key set, otherwise real OpenAI()."""
    if is_mock():
        return MockClient()
    from openai import OpenAI

    return OpenAI()


async def get_async_client() -> Any:
    """Return AsyncMockClient if no API key set, otherwise real AsyncOpenAI()."""
    if is_mock():
        return AsyncMockClient()
    from openai import AsyncOpenAI

    return AsyncOpenAI()
