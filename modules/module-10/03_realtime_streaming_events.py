"""Domain 10 - Task 10.3: Realtime Streaming Events

CONCEPTS:
  1. Event types — 25+ event types in Realtime API
  2. response.text.delta — streaming text
  3. conversation.item.created — new message
  4. response.done — response complete

Mnemonic: TRCD — Text_delta, Response, Conversation, Done

Run:
  uv run python 03_realtime_streaming_events.py
"""

from __future__ import annotations

from collections.abc import Generator
from dataclasses import dataclass, field
from enum import StrEnum

from shared.mock import get_client, is_mock

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

MODEL = "gpt-4o"
REALTIME_MODEL = "gpt-4o-realtime-preview"
NL = chr(10)

# ---------------------------------------------------------------------------
# Event type enum
# ---------------------------------------------------------------------------


class RealtimeEventType(StrEnum):
    """Key event types in the OpenAI Realtime API."""

    # Session events (server -> client)
    SESSION_CREATED = "session.created"
    SESSION_UPDATED = "session.updated"

    # Conversation events (server -> client)
    CONVERSATION_CREATED = "conversation.created"
    CONVERSATION_ITEM_CREATED = "conversation.item.created"
    CONVERSATION_ITEM_TRUNCATED = "conversation.item.truncated"
    CONVERSATION_ITEM_DELETED = "conversation.item.deleted"

    # Input audio buffer events (server -> client)
    INPUT_AUDIO_BUFFER_COMMITTED = "input_audio_buffer.committed"
    INPUT_AUDIO_BUFFER_CLEARED = "input_audio_buffer.cleared"
    INPUT_AUDIO_BUFFER_SPEECH_STARTED = "input_audio_buffer.speech_started"
    INPUT_AUDIO_BUFFER_SPEECH_STOPPED = "input_audio_buffer.speech_stopped"

    # Response lifecycle events (server -> client)
    RESPONSE_CREATED = "response.created"
    RESPONSE_DONE = "response.done"
    RESPONSE_OUTPUT_ITEM_ADDED = "response.output_item.added"
    RESPONSE_OUTPUT_ITEM_DONE = "response.output_item.done"
    RESPONSE_CONTENT_PART_ADDED = "response.content_part.added"
    RESPONSE_CONTENT_PART_DONE = "response.content_part.done"

    # Streaming delta events (server -> client)
    RESPONSE_TEXT_DELTA = "response.text.delta"
    RESPONSE_TEXT_DONE = "response.text.done"
    RESPONSE_AUDIO_TRANSCRIPT_DELTA = "response.audio_transcript.delta"
    RESPONSE_AUDIO_TRANSCRIPT_DONE = "response.audio_transcript.done"
    RESPONSE_AUDIO_DELTA = "response.audio.delta"
    RESPONSE_AUDIO_DONE = "response.audio.done"
    RESPONSE_FUNCTION_CALL_ARGUMENTS_DELTA = "response.function_call_arguments.delta"
    RESPONSE_FUNCTION_CALL_ARGUMENTS_DONE = "response.function_call_arguments.done"

    # Error event (server -> client)
    ERROR = "error"


# ---------------------------------------------------------------------------
# Event dataclasses
# ---------------------------------------------------------------------------


@dataclass
class BaseEvent:
    """Base fields present on every Realtime API event."""

    type: str
    event_id: str = "event_mock001"


@dataclass
class ConversationItemCreatedEvent(BaseEvent):
    """A new conversation item has been created."""

    type: str = RealtimeEventType.CONVERSATION_ITEM_CREATED
    item_id: str = "item_mock001"
    item_type: str = "message"
    role: str = "user"
    content: list[dict[str, str]] = field(default_factory=list)


@dataclass
class ResponseCreatedEvent(BaseEvent):
    """A response has been initiated."""

    type: str = RealtimeEventType.RESPONSE_CREATED
    response_id: str = "resp_mock001"
    status: str = "in_progress"


@dataclass
class ResponseTextDeltaEvent(BaseEvent):
    """Incremental text content in a response."""

    type: str = RealtimeEventType.RESPONSE_TEXT_DELTA
    response_id: str = "resp_mock001"
    item_id: str = "item_mock001"
    delta: str = ""


@dataclass
class ResponseTextDoneEvent(BaseEvent):
    """Text content part is complete."""

    type: str = RealtimeEventType.RESPONSE_TEXT_DONE
    response_id: str = "resp_mock001"
    item_id: str = "item_mock001"
    text: str = ""


@dataclass
class ResponseDoneEvent(BaseEvent):
    """Full response is complete."""

    type: str = RealtimeEventType.RESPONSE_DONE
    response_id: str = "resp_mock001"
    status: str = "completed"
    usage: dict[str, int] = field(
        default_factory=lambda: {
            "total_tokens": 50,
            "input_tokens": 20,
            "output_tokens": 30,
        }
    )


@dataclass
class ErrorEvent(BaseEvent):
    """An error occurred."""

    type: str = RealtimeEventType.ERROR
    code: str = "invalid_request_error"
    message: str = "An error occurred."


# Union type for all events the dispatcher handles
DispatchableEvent = (
    ConversationItemCreatedEvent
    | ResponseCreatedEvent
    | ResponseTextDeltaEvent
    | ResponseTextDoneEvent
    | ResponseDoneEvent
    | ErrorEvent
)

# ---------------------------------------------------------------------------
# Mock event dispatcher
# ---------------------------------------------------------------------------


def mock_text_conversation_stream(
    user_text: str = "What is the capital of France?",
    model_reply: str = "The capital of France is Paris.",
) -> Generator[DispatchableEvent, None, None]:
    """Yield events simulating a complete text-mode conversation turn."""
    yield ConversationItemCreatedEvent(
        event_id="event_001",
        item_id="item_user001",
        item_type="message",
        role="user",
        content=[{"type": "text", "text": user_text}],
    )
    yield ResponseCreatedEvent(event_id="event_002", response_id="resp_001")

    words = model_reply.split()
    for i, word in enumerate(words):
        yield ResponseTextDeltaEvent(
            event_id=f"event_{i + 10:03d}",
            response_id="resp_001",
            item_id="item_asst001",
            delta=word + (" " if i < len(words) - 1 else ""),
        )

    yield ResponseTextDoneEvent(
        event_id="event_090",
        response_id="resp_001",
        item_id="item_asst001",
        text=model_reply,
    )
    yield ResponseDoneEvent(
        event_id="event_099",
        response_id="resp_001",
        status="completed",
        usage={"total_tokens": 60, "input_tokens": 20, "output_tokens": 40},
    )


def dispatch_event(event: DispatchableEvent) -> str:
    """Handle a Realtime event using match/case and return a log message."""
    match event.type:
        case RealtimeEventType.CONVERSATION_ITEM_CREATED:
            assert isinstance(event, ConversationItemCreatedEvent)
            return f"[conversation.item.created] role={event.role} item_id={event.item_id}"
        case RealtimeEventType.RESPONSE_CREATED:
            assert isinstance(event, ResponseCreatedEvent)
            return f"[response.created] response_id={event.response_id} status={event.status}"
        case RealtimeEventType.RESPONSE_TEXT_DELTA:
            assert isinstance(event, ResponseTextDeltaEvent)
            return f"[response.text.delta] delta={event.delta!r}"
        case RealtimeEventType.RESPONSE_TEXT_DONE:
            assert isinstance(event, ResponseTextDoneEvent)
            return f"[response.text.done] text={event.text!r}"
        case RealtimeEventType.RESPONSE_DONE:
            assert isinstance(event, ResponseDoneEvent)
            return (
                f"[response.done] status={event.status} "
                f"tokens={event.usage.get('total_tokens', 0)}"
            )
        case RealtimeEventType.ERROR:
            assert isinstance(event, ErrorEvent)
            return f"[error] code={event.code} message={event.message}"
        case _:
            return f"[{event.type}] (unhandled)"


# ---------------------------------------------------------------------------
# Demo functions
# ---------------------------------------------------------------------------


def demo_event_types_enum() -> None:
    """DEMO 1: Show all defined Realtime API event types."""
    print(f"Realtime API defines {len(RealtimeEventType)} key event types:")
    categories = {
        "Session": [e for e in RealtimeEventType if e.value.startswith("session")],
        "Conversation": [e for e in RealtimeEventType if e.value.startswith("conversation")],
        "Audio Buffer": [e for e in RealtimeEventType if e.value.startswith("input_audio")],
        "Response": [e for e in RealtimeEventType if e.value.startswith("response")],
        "Error": [e for e in RealtimeEventType if e.value == "error"],
    }
    for category, events in categories.items():
        print(f"  {category}:")
        for evt in events:
            print(f"    {evt.value}")


def demo_event_dispatcher() -> None:
    """DEMO 2: Run event stream through the dispatcher."""
    user_q = "What is the capital of France?"
    model_a = "The capital of France is Paris."
    print(f"User: {user_q}")
    print(f"Model: {model_a}")
    print()
    print("Event dispatch log:")
    for event in mock_text_conversation_stream(user_q, model_a):
        log_line = dispatch_event(event)
        print(f"  {log_line}")


def demo_text_accumulation() -> None:
    """DEMO 3: Accumulate response.text.delta events into full text."""
    reply = "Streaming text arrives word by word from the server."
    accumulated = ""

    print("Accumulating text deltas...")
    for event in mock_text_conversation_stream("demo", reply):
        if isinstance(event, ResponseTextDeltaEvent):
            accumulated += event.delta
        elif isinstance(event, ResponseDoneEvent):
            print(f"  Final accumulated text: {accumulated!r}")
            print(f"  Response status       : {event.status}")
            print(f"  Total tokens          : {event.usage.get('total_tokens', 0)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _client = get_client()
    _mock_note = "(mock mode)" if is_mock() else "(live mode)"
    sep = "=" * 60

    print(sep)
    print(f"Domain 10 - Task 10.3: Realtime Streaming Events {_mock_note}")
    print(sep)

    print(NL + "DEMO 1: Event Types Enum")
    print("-" * 40)
    demo_event_types_enum()

    print(NL + "DEMO 2: Event Dispatcher (match/case)")
    print("-" * 40)
    demo_event_dispatcher()

    print(NL + "DEMO 3: Text Delta Accumulation")
    print("-" * 40)
    demo_text_accumulation()

    print(NL + sep)
    print("KEY TAKEAWAYS:")
    print("  - Realtime API has 25+ event types grouped by session/conversation/response/error")
    print("  - response.text.delta events stream text incrementally; accumulate into full text")
    print("  - conversation.item.created fires when a new message enters the conversation")
    print("  - response.done signals completion and includes token usage metadata")
    print("  - match/case dispatch provides a clean pattern for handling all event types")
    print(sep)
