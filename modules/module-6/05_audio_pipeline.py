"""Domain 6 - Task 6.5: STT → Process → TTS Pipeline

CONCEPTS:
  1. Speech-to-text — transcribe user audio to text
  2. Processing — send transcript through chat model
  3. Text-to-speech — convert model response to audio
  4. Round-trip latency — STT + chat + TTS latency sum

Mnemonic: SPRT — Stt, Process, Respond, Tts

Run:
  uv run python 05_audio_pipeline.py
"""

NL = chr(10)
MODEL = "gpt-4o"
WHISPER_MODEL = "whisper-1"
TTS_MODEL = "tts-1"

import io
import time

from shared.mock import get_client, is_mock


def transcribe_audio(client: object, audio_bytes: bytes) -> str:
    """Step 1: Convert audio to text."""
    audio_file = io.BytesIO(audio_bytes)
    audio_file.name = "input.mp3"
    result = client.audio.transcriptions.create(model=WHISPER_MODEL, file=audio_file)  # type: ignore[attr-defined]
    return result.text


def generate_response(client: object, transcript: str) -> str:
    """Step 2: Generate a chat response from the transcript."""
    completion = client.chat.completions.create(  # type: ignore[attr-defined]
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful voice assistant. Keep responses concise."},
            {"role": "user", "content": transcript},
        ],
    )
    return completion.choices[0].message.content or ""


def synthesize_speech(client: object, text: str) -> bytes:
    """Step 3: Convert response text back to audio."""
    response = client.audio.speech.create(model=TTS_MODEL, voice="nova", input=text)  # type: ignore[attr-defined]
    return response.content


def run_voice_pipeline(audio_bytes: bytes) -> dict[str, object]:
    """Run the complete STT → Chat → TTS pipeline."""
    client = get_client()
    results: dict[str, object] = {}

    t0 = time.perf_counter()
    transcript = transcribe_audio(client, audio_bytes)
    t1 = time.perf_counter()
    results["transcript"] = transcript
    results["stt_ms"] = round((t1 - t0) * 1000, 1)

    response_text = generate_response(client, transcript)
    t2 = time.perf_counter()
    results["response"] = response_text
    results["chat_ms"] = round((t2 - t1) * 1000, 1)

    audio_out = synthesize_speech(client, response_text)
    t3 = time.perf_counter()
    results["audio_bytes"] = len(audio_out)
    results["tts_ms"] = round((t3 - t2) * 1000, 1)
    results["total_ms"] = round((t3 - t0) * 1000, 1)

    return results


def demo_voice_pipeline() -> None:
    """DEMO 1: Run the complete voice pipeline end-to-end."""
    mock_user_audio = b"mock_user_speech_bytes_what_is_the_weather"
    print("  Input: mock user audio (simulating speech)")

    results = run_voice_pipeline(mock_user_audio)
    print(f"  Step 1 - Transcript: {results['transcript']!r}")
    print(f"           STT latency: {results['stt_ms']}ms")
    print(f"  Step 2 - Response:   {results['response']!r}")
    print(f"           Chat latency: {results['chat_ms']}ms")
    print(f"  Step 3 - Audio out:  {results['audio_bytes']} bytes")
    print(f"           TTS latency: {results['tts_ms']}ms")
    print(f"  Total round-trip:    {results['total_ms']}ms")


def demo_latency_considerations() -> None:
    """DEMO 2: Show latency breakdown and optimization strategies."""
    print("  Typical production latencies (rough estimates):")
    print("    STT (Whisper):     200-800ms depending on audio length")
    print("    Chat (gpt-4o):     300-1500ms depending on response length")
    print("    TTS (tts-1):       100-500ms depending on text length")
    print("    Total P50:         600ms-2800ms")
    print(f"{NL}  Optimization strategies:")
    print("    - Stream TTS output for time-to-first-byte improvement")
    print("    - Use gpt-4o-mini for chat to reduce latency")
    print("    - Cache common TTS responses (greetings, confirmations)")
    print("    - Run STT immediately on audio end, don't wait for silence")


if __name__ == "__main__":
    sep = "=" * 60
    mode = "MOCK" if is_mock() else "LIVE"
    print(f"{NL}{sep}{NL}Domain 6 - Task 6.5: Audio Pipeline [{mode}]{NL}{sep}")

    print(f"{NL}--- DEMO 1: Full Pipeline ---")
    demo_voice_pipeline()

    print(f"{NL}--- DEMO 2: Latency Considerations ---")
    demo_latency_considerations()

    print(f"{NL}KEY TAKEAWAYS:")
    print("  1. STT → Chat → TTS is the canonical voice assistant pattern")
    print("  2. Track latency for each step — the bottleneck varies by use case")
    print("  3. Use gpt-4o-mini for lower latency; gpt-4o for quality")
    print("  4. Streaming TTS reduces perceived latency significantly for long responses")
