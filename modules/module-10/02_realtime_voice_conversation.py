"""Domain 10 - Task 10.2: Realtime Voice Conversation

CONCEPTS:
  1. input_audio_buffer.append — send audio chunks
  2. response.audio.delta — receive audio output
  3. Turn detection — VAD (voice activity detection)
  4. Conversation flow — user speaks, model responds

Mnemonic: AVTR — Audio, Vad, Turn_detection, Response

Run:
  uv run python 02_realtime_voice_conversation.py
"""

from __future__ import annotations

import base64
from collections.abc import Generator
from dataclasses import dataclass, field

from shared.mock import get_client, is_mock

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

MODEL = "gpt-4o"
REALTIME_MODEL = "gpt-4o-realtime-preview"
NL = chr(10)

# ---------------------------------------------------------------------------
# Client-to-server event dataclasses
# ---------------------------------------------------------------------------


@dataclass
class AudioAppendEvent:
    """Client sends audio chunks to the input buffer."""

    type: str = "input_audio_buffer.append"
    audio: str = ""  # base64-encoded PCM16 audio data

    def to_dict(self) -> dict[str, object]:
        return {"type": self.type, "audio": self.audio}


@dataclass
class AudioCommitEvent:
    """Client signals end of audio input turn."""

    type: str = "input_audio_buffer.commit"

    def to_dict(self) -> dict[str, object]:
        return {"type": self.type}


@dataclass
class ResponseCreateEvent:
    """Client requests model to generate a response."""

    type: str = "response.create"
    modalities: list[str] = field(default_factory=lambda: ["audio", "text"])

    def to_dict(self) -> dict[str, object]:
        return {
            "type": self.type,
            "response": {"modalities": self.modalities},
        }


# ---------------------------------------------------------------------------
# Server-to-client event dataclasses
# ---------------------------------------------------------------------------


@dataclass
class SpeechStartedEvent:
    """VAD detected user started speaking."""

    type: str = "input_audio_buffer.speech_started"
    event_id: str = "event_mock010"
    audio_start_ms: int = 0


@dataclass
class SpeechStoppedEvent:
    """VAD detected user stopped speaking."""

    type: str = "input_audio_buffer.speech_stopped"
    event_id: str = "event_mock011"
    audio_end_ms: int = 1200


@dataclass
class ResponseAudioDeltaEvent:
    """Server streams audio output chunk."""

    type: str = "response.audio.delta"
    event_id: str = "event_mock020"
    response_id: str = "resp_mock001"
    item_id: str = "item_mock001"
    delta: str = ""  # base64-encoded PCM16 audio chunk


@dataclass
class ResponseAudioTranscriptDeltaEvent:
    """Server streams partial transcript of audio output."""

    type: str = "response.audio_transcript.delta"
    event_id: str = "event_mock021"
    response_id: str = "resp_mock001"
    item_id: str = "item_mock001"
    delta: str = ""


@dataclass
class ResponseAudioDoneEvent:
    """Server signals audio output is complete."""

    type: str = "response.audio.done"
    event_id: str = "event_mock030"
    response_id: str = "resp_mock001"
    item_id: str = "item_mock001"


@dataclass
class ResponseDoneEvent:
    """Server signals full response is complete."""

    type: str = "response.done"
    event_id: str = "event_mock031"
    response_id: str = "resp_mock001"
    status: str = "completed"


# Union type for all server events in a voice turn
ServerVoiceEvent = (
    SpeechStartedEvent
    | SpeechStoppedEvent
    | ResponseAudioDeltaEvent
    | ResponseAudioTranscriptDeltaEvent
    | ResponseAudioDoneEvent
    | ResponseDoneEvent
)

# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------


def mock_audio_chunk(text: str) -> str:
    """Return a mock base64-encoded audio chunk (simulated PCM16 bytes)."""
    fake_pcm16 = text.encode("utf-8")
    return base64.b64encode(fake_pcm16).decode("utf-8")


def build_audio_append_events(text: str) -> list[AudioAppendEvent]:
    """Split mock audio into multiple append chunks."""
    words = text.split()
    events: list[AudioAppendEvent] = []
    for word in words:
        events.append(AudioAppendEvent(audio=mock_audio_chunk(word)))
    return events


def mock_voice_conversation_stream(
    transcript: str = "Hello, how are you today?",
) -> Generator[ServerVoiceEvent, None, None]:
    """Simulate a complete voice turn: VAD detection + audio response."""
    yield SpeechStartedEvent(audio_start_ms=0)
    yield SpeechStoppedEvent(audio_end_ms=1200)

    words = transcript.split()
    for i, word in enumerate(words):
        chunk = mock_audio_chunk(word)
        yield ResponseAudioDeltaEvent(
            event_id=f"event_audio_delta_{i:03d}",
            delta=chunk,
        )
        yield ResponseAudioTranscriptDeltaEvent(
            event_id=f"event_transcript_delta_{i:03d}",
            delta=word + (" " if i < len(words) - 1 else ""),
        )

    yield ResponseAudioDoneEvent()
    yield ResponseDoneEvent()


# ---------------------------------------------------------------------------
# Demo functions
# ---------------------------------------------------------------------------


def demo_audio_append() -> None:
    """DEMO 1: Show input_audio_buffer.append event structure."""
    events = build_audio_append_events("hello world")
    print(f"Sending {len(events)} audio chunk(s) to input buffer:")
    for i, evt in enumerate(events):
        decoded = base64.b64decode(evt.audio).decode("utf-8")
        print(f"  chunk {i + 1}: type={evt.type}  audio=<base64({decoded!r})>")

    commit = AudioCommitEvent()
    print(f"  commit: type={commit.type}")


def demo_response_create() -> None:
    """DEMO 2: Show response.create event."""
    event = ResponseCreateEvent(modalities=["audio", "text"])
    payload = event.to_dict()
    print("response.create payload:")
    print(f"  type       : {payload['type']}")
    print(f"  modalities : {payload['response']['modalities']}")  # type: ignore[index]


def demo_voice_conversation_stream() -> None:
    """DEMO 3: Simulate receiving the full voice conversation event stream."""
    response_text = "I am doing well, thank you for asking."
    print(f"Simulating voice response for: {response_text!r}")
    print()

    accumulated_transcript = ""
    audio_chunk_count = 0

    for event in mock_voice_conversation_stream(response_text):
        if isinstance(event, SpeechStartedEvent):
            print(f"  [{event.type}] audio_start_ms={event.audio_start_ms}")
        elif isinstance(event, SpeechStoppedEvent):
            print(f"  [{event.type}] audio_end_ms={event.audio_end_ms}")
        elif isinstance(event, ResponseAudioDeltaEvent):
            audio_chunk_count += 1
        elif isinstance(event, ResponseAudioTranscriptDeltaEvent):
            accumulated_transcript += event.delta
        elif isinstance(event, ResponseAudioDoneEvent):
            print(f"  [{event.type}] total audio chunks received: {audio_chunk_count}")
        elif isinstance(event, ResponseDoneEvent):
            print(f"  [{event.type}] status={event.status}")

    print(f"  Accumulated transcript: {accumulated_transcript!r}")


def demo_vad_modes() -> None:
    """DEMO 4: Show VAD turn detection modes."""
    server_vad = {
        "type": "server_vad",
        "threshold": 0.5,
        "prefix_padding_ms": 300,
        "silence_duration_ms": 500,
        "create_response": True,
    }
    no_vad: dict[str, object] = {"type": "none"}

    print("Server VAD config (automatic turn detection):")
    for key, value in server_vad.items():
        print(f"  {key}: {value}")
    print()
    print("No VAD config (manual turn control):")
    for key, value in no_vad.items():
        print(f"  {key}: {value}")
    print("  -> Client must send input_audio_buffer.commit manually")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _client = get_client()
    _mock_note = "(mock mode)" if is_mock() else "(live mode)"
    sep = "=" * 60

    print(sep)
    print(f"Domain 10 - Task 10.2: Realtime Voice Conversation {_mock_note}")
    print(sep)

    print(NL + "DEMO 1: Audio Buffer Append")
    print("-" * 40)
    demo_audio_append()

    print(NL + "DEMO 2: Response Create Event")
    print("-" * 40)
    demo_response_create()

    print(NL + "DEMO 3: Full Voice Conversation Event Stream")
    print("-" * 40)
    demo_voice_conversation_stream()

    print(NL + "DEMO 4: VAD Turn Detection Modes")
    print("-" * 40)
    demo_vad_modes()

    print(NL + sep)
    print("KEY TAKEAWAYS:")
    print("  - Audio is sent as base64-encoded PCM16 chunks via input_audio_buffer.append")
    print("  - Server VAD fires speech_started/stopped events automatically on voice activity")
    print("  - response.audio.delta and response.audio_transcript.delta stream in parallel")
    print("  - Without VAD (type=none) the client controls turns via input_audio_buffer.commit")
    print("  - response.done marks the end of a complete model turn with status=completed")
    print(sep)
