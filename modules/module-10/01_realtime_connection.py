"""Domain 10 - Task 10.1: Realtime API Connection Setup

CONCEPTS:
  1. WebSocket-based — persistent bidirectional connection
  2. Session config — model, voice, modalities
  3. Event types — session.created, session.updated
  4. API endpoint — wss://api.openai.com/v1/realtime

Mnemonic: WSME — Websocket, Session, Modalities, Events

Run:
  uv run python 01_realtime_connection.py
"""

from __future__ import annotations

from collections.abc import Generator
from dataclasses import dataclass, field

from shared.mock import get_client, is_mock

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

MODEL = "gpt-4o"
REALTIME_MODEL = "gpt-4o-realtime-preview"
REALTIME_ENDPOINT = "wss://api.openai.com/v1/realtime"
NL = chr(10)

# ---------------------------------------------------------------------------
# Event dataclasses
# ---------------------------------------------------------------------------


@dataclass
class SessionConfig:
    """Configuration sent in session.create or session.update."""

    model: str = REALTIME_MODEL
    voice: str = "alloy"
    modalities: list[str] = field(default_factory=lambda: ["audio", "text"])
    instructions: str = "You are a helpful assistant."
    turn_detection: dict[str, object] = field(
        default_factory=lambda: {
            "type": "server_vad",
            "threshold": 0.5,
            "prefix_padding_ms": 300,
            "silence_duration_ms": 500,
        }
    )


@dataclass
class SessionCreateEvent:
    """Client-to-server event to create a new session."""

    type: str = "session.create"
    session: SessionConfig = field(default_factory=SessionConfig)

    def to_dict(self) -> dict[str, object]:
        return {
            "type": self.type,
            "session": {
                "model": self.session.model,
                "voice": self.session.voice,
                "modalities": self.session.modalities,
                "instructions": self.session.instructions,
                "turn_detection": self.session.turn_detection,
            },
        }


@dataclass
class SessionCreatedEvent:
    """Server-to-client event confirming session creation."""

    type: str = "session.created"
    event_id: str = "event_mock001"
    session_id: str = "sess_mock001"
    model: str = REALTIME_MODEL
    voice: str = "alloy"
    modalities: list[str] = field(default_factory=lambda: ["audio", "text"])


@dataclass
class SessionUpdatedEvent:
    """Server-to-client event confirming session update."""

    type: str = "session.updated"
    event_id: str = "event_mock002"
    session_id: str = "sess_mock001"
    model: str = REALTIME_MODEL
    voice: str = "alloy"
    modalities: list[str] = field(default_factory=lambda: ["audio", "text"])


# ---------------------------------------------------------------------------
# Mock connection helpers
# ---------------------------------------------------------------------------


def build_connection_url(model: str = REALTIME_MODEL) -> str:
    """Return the WebSocket URL for the Realtime API."""
    return f"{REALTIME_ENDPOINT}?model={model}"


def mock_event_stream(
    config: SessionConfig,
) -> Generator[SessionCreatedEvent | SessionUpdatedEvent, None, None]:
    """Simulate server events after a session.create handshake."""
    yield SessionCreatedEvent(
        model=config.model,
        voice=config.voice,
        modalities=config.modalities,
    )
    yield SessionUpdatedEvent(
        model=config.model,
        voice=config.voice,
        modalities=config.modalities,
    )


# ---------------------------------------------------------------------------
# Demo functions
# ---------------------------------------------------------------------------


def demo_session_config() -> None:
    """DEMO 1: Show session configuration structure."""
    config = SessionConfig()
    create_event = SessionCreateEvent(session=config)
    payload = create_event.to_dict()

    print("Session config (client -> server):")
    for key, value in payload["session"].items():  # type: ignore[union-attr]
        print(f"  {key}: {value}")


def demo_connection_url() -> None:
    """DEMO 2: Show WebSocket endpoint URL."""
    url = build_connection_url()
    print(f"WebSocket endpoint: {url}")
    print("Protocol: WSS (TLS-encrypted WebSocket)")
    print("Auth: Authorization: Bearer <OPENAI_API_KEY> header")
    print("Header: OpenAI-Beta: realtime=v1")


def demo_event_stream(config: SessionConfig) -> None:
    """DEMO 3: Simulate receiving server events after connection."""
    print("Simulating server event stream...")
    for event in mock_event_stream(config):
        if isinstance(event, SessionCreatedEvent):
            print(f"  Received: {event.type}")
            print(f"    session_id : {event.session_id}")
            print(f"    model      : {event.model}")
            print(f"    voice      : {event.voice}")
            print(f"    modalities : {event.modalities}")
        elif isinstance(event, SessionUpdatedEvent):
            print(f"  Received: {event.type}")
            print(f"    session_id : {event.session_id}")


def anti_pattern_websockets_import() -> None:
    """ANTI-PATTERN 1: Importing external WebSocket libraries.

    The following would require an external dependency and is NOT the
    approach used here:

        import websockets  # WRONG — external dependency not in project
        async with websockets.connect(url) as ws:
            ...

    Use mock simulation or the OpenAI SDK's built-in realtime client
    (openai.AsyncOpenAI().beta.realtime) in production instead.
    """
    print("ANTI-PATTERN: Do NOT import 'websockets' or 'websocket-client'.")
    print("  Use OpenAI SDK realtime client in production code.")
    print("  Use mock simulation (generators) for learning/testing.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _client = get_client()
    _mock_note = "(mock mode)" if is_mock() else "(live mode)"
    sep = "=" * 60

    print(sep)
    print(f"Domain 10 - Task 10.1: Realtime API Connection Setup {_mock_note}")
    print(sep)

    print(NL + "DEMO 1: Session Configuration")
    print("-" * 40)
    demo_session_config()

    print(NL + "DEMO 2: WebSocket Connection URL")
    print("-" * 40)
    demo_connection_url()

    print(NL + "DEMO 3: Mock Event Stream")
    print("-" * 40)
    _config = SessionConfig()
    demo_event_stream(_config)

    print(NL + "ANTI-PATTERN 1: External WebSocket Libraries")
    print("-" * 40)
    anti_pattern_websockets_import()

    print(NL + sep)
    print("KEY TAKEAWAYS:")
    print("  - Realtime API uses WSS (WebSocket Secure) for persistent bidirectional I/O")
    print("  - session.create is the first client->server event; server replies session.created")
    print("  - Session config sets model, voice, modalities, instructions, and turn_detection")
    print("  - The endpoint URL encodes the model: wss://api.openai.com/v1/realtime?model=...")
    print("  - Mock simulation with generators replicates the event-driven flow without WSS")
    print(sep)
