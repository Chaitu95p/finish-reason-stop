"""Domain 10 - Task 10.5: Realtime Session Management

CONCEPTS:
  1. session.update — change modalities, voice, tools mid-session
  2. VAD configuration — threshold, silence_duration_ms
  3. Modalities — audio + text vs text only
  4. Session cleanup — proper connection close

Mnemonic: UVMC — Update, Vad, Modalities, Cleanup

Run:
  uv run python 05_realtime_session_management.py
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
# Enums
# ---------------------------------------------------------------------------


class ModalityMode(StrEnum):
    """Supported modality combinations for a Realtime session."""

    AUDIO_AND_TEXT = "audio_and_text"
    TEXT_ONLY = "text_only"


class VoiceOption(StrEnum):
    """Available voices for audio output."""

    ALLOY = "alloy"
    ECHO = "echo"
    FABLE = "fable"
    ONYX = "onyx"
    NOVA = "nova"
    SHIMMER = "shimmer"


# ---------------------------------------------------------------------------
# VAD configuration dataclass
# ---------------------------------------------------------------------------


@dataclass
class ServerVadConfig:
    """Configuration for server-side Voice Activity Detection."""

    type: str = "server_vad"
    threshold: float = 0.5
    prefix_padding_ms: int = 300
    silence_duration_ms: int = 500
    create_response: bool = True

    def to_dict(self) -> dict[str, object]:
        return {
            "type": self.type,
            "threshold": self.threshold,
            "prefix_padding_ms": self.prefix_padding_ms,
            "silence_duration_ms": self.silence_duration_ms,
            "create_response": self.create_response,
        }


@dataclass
class NoVadConfig:
    """Disable VAD — client controls turn timing manually."""

    type: str = "none"

    def to_dict(self) -> dict[str, object]:
        return {"type": self.type}


# ---------------------------------------------------------------------------
# Session update event dataclass
# ---------------------------------------------------------------------------


@dataclass
class SessionUpdateEvent:
    """Client-to-server event to modify session configuration mid-session."""

    type: str = "session.update"
    modalities: list[str] = field(default_factory=lambda: ["audio", "text"])
    voice: str = VoiceOption.ALLOY
    instructions: str = "You are a helpful assistant."
    turn_detection: dict[str, object] = field(
        default_factory=lambda: ServerVadConfig().to_dict()
    )
    tools: list[dict[str, object]] = field(default_factory=list)
    tool_choice: str = "auto"
    temperature: float = 0.8
    max_response_output_tokens: int | str = 4096

    def to_dict(self) -> dict[str, object]:
        return {
            "type": self.type,
            "session": {
                "modalities": self.modalities,
                "voice": self.voice.value if isinstance(self.voice, VoiceOption) else self.voice,
                "instructions": self.instructions,
                "turn_detection": self.turn_detection,
                "tools": self.tools,
                "tool_choice": self.tool_choice,
                "temperature": self.temperature,
                "max_response_output_tokens": self.max_response_output_tokens,
            },
        }


@dataclass
class SessionUpdatedEvent:
    """Server-to-client confirmation of session update."""

    type: str = "session.updated"
    event_id: str = "event_mock060"
    session_id: str = "sess_mock001"
    modalities: list[str] = field(default_factory=lambda: ["audio", "text"])
    voice: str = VoiceOption.ALLOY


@dataclass
class ResponseCancelEvent:
    """Client-to-server event to cancel an in-flight response."""

    type: str = "response.cancel"
    response_id: str | None = None

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {"type": self.type}
        if self.response_id is not None:
            payload["response_id"] = self.response_id
        return payload


# ---------------------------------------------------------------------------
# Session lifecycle simulator
# ---------------------------------------------------------------------------


@dataclass
class MockRealtimeSession:
    """Simulates lifecycle state of a Realtime session."""

    session_id: str = "sess_mock001"
    modalities: list[str] = field(default_factory=lambda: ["audio", "text"])
    voice: str = VoiceOption.ALLOY
    connected: bool = False
    in_flight_response_id: str | None = None

    def connect(self) -> None:
        """Simulate opening the WebSocket connection."""
        self.connected = True

    def disconnect(self) -> None:
        """Simulate closing the WebSocket connection."""
        self.connected = False

    def is_connected(self) -> bool:
        return self.connected

    def apply_update(self, event: SessionUpdateEvent) -> SessionUpdatedEvent:
        """Apply a session.update and return the server confirmation."""
        self.modalities = event.modalities
        voice_val = event.voice.value if isinstance(event.voice, VoiceOption) else event.voice
        self.voice = voice_val
        return SessionUpdatedEvent(
            session_id=self.session_id,
            modalities=self.modalities,
            voice=voice_val,
        )

    def cancel_response(self, response_id: str | None = None) -> dict[str, object]:
        """Build and apply a response.cancel event."""
        rid = response_id or self.in_flight_response_id
        cancel = ResponseCancelEvent(response_id=rid)
        self.in_flight_response_id = None
        return cancel.to_dict()


# ---------------------------------------------------------------------------
# Mock event generators
# ---------------------------------------------------------------------------


def mock_session_update_stream(
    update_event: SessionUpdateEvent,
) -> Generator[SessionUpdatedEvent, None, None]:
    """Simulate server acknowledging session.update."""
    yield SessionUpdatedEvent(
        modalities=update_event.modalities,
        voice=update_event.voice,
    )


# ---------------------------------------------------------------------------
# Demo functions
# ---------------------------------------------------------------------------


def demo_session_update_structure() -> None:
    """DEMO 1: Show session.update event structure."""
    vad = ServerVadConfig(threshold=0.6, silence_duration_ms=400)
    update = SessionUpdateEvent(
        modalities=["audio", "text"],
        voice=VoiceOption.NOVA,
        turn_detection=vad.to_dict(),
        temperature=0.7,
        max_response_output_tokens=2048,
    )
    payload = update.to_dict()
    session = payload["session"]

    print("session.update payload:")
    print(f"  type          : {payload['type']}")
    print(f"  modalities    : {session['modalities']}")  # type: ignore[index]
    print(f"  voice         : {session['voice']}")  # type: ignore[index]
    print(f"  temperature   : {session['temperature']}")  # type: ignore[index]
    print(f"  max_output_tok: {session['max_response_output_tokens']}")  # type: ignore[index]
    print(f"  turn_detection: {session['turn_detection']}")  # type: ignore[index]


def demo_vad_configuration() -> None:
    """DEMO 2: Show VAD configuration options."""
    server_vad = ServerVadConfig(
        threshold=0.5,
        prefix_padding_ms=300,
        silence_duration_ms=500,
        create_response=True,
    )
    no_vad = NoVadConfig()

    print("Server VAD (automatic turn detection):")
    for key, value in server_vad.to_dict().items():
        print(f"  {key}: {value}")
    print()
    print("  threshold          : 0.0-1.0, higher = less sensitive")
    print("  prefix_padding_ms  : audio kept before speech start")
    print("  silence_duration_ms: silence needed to end a turn")
    print("  create_response    : auto-trigger response on turn end")
    print()
    print("No VAD (manual control):")
    for key, value in no_vad.to_dict().items():
        print(f"  {key}: {value}")
    print("  -> Use input_audio_buffer.commit to end turns manually")


def demo_modality_switching() -> None:
    """DEMO 3: Show how to switch between modality modes mid-session."""
    session = MockRealtimeSession()
    session.connect()

    print(f"Session connected: {session.is_connected()}")
    print(f"Initial modalities: {session.modalities}")

    # Switch to text-only
    text_only_update = SessionUpdateEvent(
        modalities=["text"],
        voice=VoiceOption.ALLOY,
    )
    confirmed = session.apply_update(text_only_update)
    print()
    print("After switch to text-only:")
    print(f"  sent    : {text_only_update.type}  modalities={text_only_update.modalities}")
    print(f"  received: {confirmed.type}  modalities={confirmed.modalities}")

    # Switch back to audio + text
    audio_update = SessionUpdateEvent(
        modalities=["audio", "text"],
        voice=VoiceOption.NOVA,
    )
    confirmed2 = session.apply_update(audio_update)
    print()
    print("After switch back to audio+text:")
    print(f"  sent    : {audio_update.type}  modalities={audio_update.modalities}")
    print(f"  received: {confirmed2.type}  modalities={confirmed2.modalities}  voice={confirmed2.voice}")


def demo_cleanup_pattern() -> None:
    """DEMO 4: Demonstrate proper session cleanup."""
    session = MockRealtimeSession(in_flight_response_id="resp_active001")
    session.connect()

    print("Session state before cleanup:")
    print(f"  connected           : {session.is_connected()}")
    print(f"  in_flight_response  : {session.in_flight_response_id}")

    # Step 1: cancel any in-flight response
    cancel_payload = session.cancel_response()
    print()
    print("Step 1 — Cancel in-flight response:")
    print(f"  sending: {cancel_payload}")

    # Step 2: disconnect
    session.disconnect()
    print()
    print("Step 2 — Close WebSocket connection:")
    print("  session.disconnect() called")
    print(f"  connected: {session.is_connected()}")
    print()
    print("  In production: await ws.close() or ws.close() on the WebSocket object")
    print("  Resources freed: audio buffers, conversation state, server-side session")


def demo_voice_options() -> None:
    """DEMO 5: List available voice options."""
    print("Available voices for Realtime API:")
    for voice in VoiceOption:
        marker = " (default)" if voice == VoiceOption.ALLOY else ""
        print(f"  {voice.value}{marker}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _client = get_client()
    _mock_note = "(mock mode)" if is_mock() else "(live mode)"
    sep = "=" * 60

    print(sep)
    print(f"Domain 10 - Task 10.5: Realtime Session Management {_mock_note}")
    print(sep)

    print(NL + "DEMO 1: session.update Structure")
    print("-" * 40)
    demo_session_update_structure()

    print(NL + "DEMO 2: VAD Configuration Options")
    print("-" * 40)
    demo_vad_configuration()

    print(NL + "DEMO 3: Modality Switching Mid-Session")
    print("-" * 40)
    demo_modality_switching()

    print(NL + "DEMO 4: Proper Session Cleanup")
    print("-" * 40)
    demo_cleanup_pattern()

    print(NL + "DEMO 5: Available Voices")
    print("-" * 40)
    demo_voice_options()

    print(NL + sep)
    print("KEY TAKEAWAYS:")
    print("  - session.update can change modalities, voice, tools, VAD config, and instructions mid-session")
    print("  - Server VAD (type=server_vad) handles turn detection automatically via silence detection")
    print("  - Switching to text-only modalities ([\"text\"]) disables audio input and output entirely")
    print("  - Cleanup order: cancel in-flight response -> close WebSocket -> release resources")
    print("  - max_response_output_tokens can be an int or the string 'inf' for unlimited output")
    print(sep)
