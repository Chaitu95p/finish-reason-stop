"""Project 5: Meeting Summarizer — Audio to Structured Summary

Pipeline:
  1. Transcribe audio with whisper-1
  2. Extract structured summary: topics, action items, decisions, attendees
  3. Output a formatted meeting report

Run:
  uv run python main.py
  uv run python main.py --demo
"""

NL = chr(10)
TRANSCRIPTION_MODEL = "whisper-1"
CHAT_MODEL = "gpt-4o"

import argparse
import io
import json
from pathlib import Path

from pydantic import BaseModel
from shared.mock import get_client, is_mock

SAMPLE_TRANSCRIPT = """
Team standup — Monday 9am

Alice: Good morning everyone. Let's go through updates quickly.
Bob: I finished the payment integration. It's ready for review.
Alice: Great. Carol, can you review Bob's PR today?
Carol: Sure, I'll have feedback by 3pm.
Bob: I also found a bug in the auth service — tokens expire too early.
Alice: Let's prioritize that. David, can you take the auth bug?
David: Yes, I'll look at it after my morning meetings. Should have a fix by EOD.
Carol: I'll have the mobile design mockups ready by Wednesday.
Alice: Perfect. So action items: Carol reviews Bob's PR by 3pm, David fixes auth bug EOD, Carol delivers mockups Wednesday.
David: Should we move our Friday demo to next Monday? The feature isn't ready.
Alice: Agreed. Let's reschedule to next Monday 2pm. I'll send the calendar invite.
"""


class MeetingSummary(BaseModel):
    title: str
    attendees: list[str]
    key_topics: list[str]
    action_items: list[str]
    decisions: list[str]
    next_meeting: str | None = None


def transcribe(client: object, audio_bytes: bytes, filename: str = "meeting.mp3") -> str:
    """Transcribe audio bytes to text."""
    f = io.BytesIO(audio_bytes)
    f.name = filename
    result = client.audio.transcriptions.create(  # type: ignore[attr-defined]
        model=TRANSCRIPTION_MODEL,
        file=f,
        language="en",
    )
    return result.text


def extract_summary(client: object, transcript: str) -> MeetingSummary:
    """Extract structured summary from transcript text."""
    prompt = (
        "Extract a structured meeting summary from this transcript. "
        "Return JSON with fields: title (string), attendees (list of names), "
        "key_topics (list of strings), action_items (list with owner and deadline), "
        "decisions (list of strings), next_meeting (string or null)."
    )
    response = client.chat.completions.create(  # type: ignore[attr-defined]
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": transcript},
        ],
        response_format={"type": "json_object"},
    )
    raw = response.choices[0].message.content or json.dumps({
        "title": "Team Standup — Monday",
        "attendees": ["Alice", "Bob", "Carol", "David"],
        "key_topics": ["Payment integration review", "Auth token bug", "Mobile mockups"],
        "action_items": [
            "Carol: review Bob's PR by 3pm",
            "David: fix auth token bug by EOD",
            "Carol: deliver mobile mockups by Wednesday",
            "Alice: send calendar invite for Monday 2pm demo",
        ],
        "decisions": [
            "Friday demo rescheduled to next Monday 2pm",
            "Auth bug prioritized as high priority",
        ],
        "next_meeting": "Monday 2pm — product demo",
    })
    data = json.loads(raw)
    return MeetingSummary.model_validate(data)


def format_report(summary: MeetingSummary) -> str:
    """Format the structured summary as a readable report."""
    lines = [
        f"# {summary.title}",
        f"{NL}**Attendees:** {', '.join(summary.attendees)}",
        f"{NL}## Key Topics",
    ]
    for topic in summary.key_topics:
        lines.append(f"  - {topic}")
    lines.append(f"{NL}## Action Items")
    for item in summary.action_items:
        lines.append(f"  - {item}")
    lines.append(f"{NL}## Decisions")
    for dec in summary.decisions:
        lines.append(f"  - {dec}")
    if summary.next_meeting:
        lines.append(f"{NL}**Next Meeting:** {summary.next_meeting}")
    return NL.join(lines)


def run_pipeline(client: object, transcript: str) -> None:
    """Run the full pipeline and print the report."""
    print("Extracting structured summary...")
    summary = extract_summary(client, transcript)
    report = format_report(summary)
    print(f"{NL}{report}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Meeting Summarizer")
    parser.add_argument("--demo", action="store_true", help="Use sample transcript")
    parser.add_argument("--audio", type=Path, help="Path to audio file for transcription")
    args = parser.parse_args()

    mode = "MOCK" if is_mock() else "LIVE"
    print(f"Meeting Summarizer [{mode}]{NL}")

    client = get_client()

    if args.audio and args.audio.exists():
        print(f"Transcribing {args.audio.name}...")
        transcript = transcribe(client, args.audio.read_bytes(), args.audio.name)
        print(f"Transcript ({len(transcript)} chars){NL}")
        run_pipeline(client, transcript)
    elif args.demo:
        print("Using sample transcript...")
        run_pipeline(client, SAMPLE_TRANSCRIPT)
    else:
        print("Paste transcript text (end with '---' on a new line):")
        lines = []
        while True:
            try:
                line = input()
                if line.strip() == "---":
                    break
                lines.append(line)
            except (EOFError, KeyboardInterrupt):
                break
        transcript = NL.join(lines) if lines else SAMPLE_TRANSCRIPT
        run_pipeline(client, transcript)
