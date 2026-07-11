"""Domain 1 - Task 1.1: Responses API Basics

CONCEPTS:
  1. client.responses.create() — the primary OpenAI Responses API
  2. response.output_text — convenience property for plain text
  3. response.status — always check "completed" before using output
  4. response.output — list of typed output items

Mnemonic: SORT — Status, Output, Response_id, Text

Run:
  uv run python 01_responses_api_basics.py
"""

NL = chr(10)
MODEL = "gpt-4o"

from shared.mock import get_client, is_mock


def demo_basic_response() -> None:
    """DEMO 1: Create a response and access the text output."""
    client = get_client()
    response = client.responses.create(
        model=MODEL,
        input="What is the capital of France?",
    )
    print(f"  Status:      {response.status}")
    print(f"  Output text: {response.output_text}")
    print(f"  Response ID: {response.id}")


def demo_output_items() -> None:
    """DEMO 2: Iterate over response.output items explicitly."""
    client = get_client()
    response = client.responses.create(
        model=MODEL,
        input="Give me three Python tips.",
    )
    print(f"  Output items ({len(response.output)} total):")
    for i, item in enumerate(response.output):
        print(f"    [{i}] type={item.type!r}  text={item.text!r}")


def demo_with_instructions() -> None:
    """DEMO 3: Pass a system-level instruction via the instructions param."""
    client = get_client()
    response = client.responses.create(
        model=MODEL,
        instructions="You are a concise assistant. Reply in one sentence.",
        input="Explain quantum entanglement.",
    )
    print(f"  Instructed response: {response.output_text!r}")


def anti_pattern_ignore_status() -> None:
    """ANTI-PATTERN 1: Using output without checking status.

    If status != 'completed' the output list may be empty,
    causing an IndexError on output[0].
    """
    client = get_client()
    response = client.responses.create(model=MODEL, input="Hello")

    # WRONG:
    # text = response.output[0].text   # IndexError if output is empty!

    # CORRECT: always guard on status
    if response.status == "completed" and response.output:
        text = response.output[0].text
        print(f"  Safe access: {text!r}")
    else:
        print(f"  Unexpected status: {response.status}")


if __name__ == "__main__":
    sep = "=" * 60
    mode = "MOCK" if is_mock() else "LIVE"
    print(f"{NL}{sep}{NL}Domain 1 - Task 1.1: Responses API Basics [{mode}]{NL}{sep}")

    print(f"{NL}--- DEMO 1: Basic Response ---")
    demo_basic_response()

    print(f"{NL}--- DEMO 2: Output Items ---")
    demo_output_items()

    print(f"{NL}--- DEMO 3: With Instructions ---")
    demo_with_instructions()

    print(f"{NL}--- ANTI-PATTERN 1: Ignore Status ---")
    anti_pattern_ignore_status()

    print(f"{NL}KEY TAKEAWAYS:")
    print("  1. Use response.output_text for simple text — it's a convenience shortcut")
    print("  2. Always check response.status == 'completed' before accessing output")
    print("  3. response.output is a list; iterate it for multi-part or tool-call outputs")
    print("  4. response.id enables multi-turn via previous_response_id (see Task 1.5)")
