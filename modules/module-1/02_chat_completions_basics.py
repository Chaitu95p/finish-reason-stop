"""Domain 1 - Task 1.2: Chat Completions Basics

CONCEPTS:
  1. client.chat.completions.create() — the classic Chat Completions API
  2. messages list — system/user/assistant role pattern
  3. choices[0].message.content — accessing the response text
  4. finish_reason — "stop" means natural completion, always check it

Mnemonic: CMFC — Create, Messages, Finish_reason, Content

Run:
  uv run python 02_chat_completions_basics.py
"""

NL = chr(10)
MODEL = "gpt-4o"

from shared.mock import get_client, is_mock


def demo_basic_chat() -> None:
    """DEMO 1: Minimal chat completion with a user message."""
    client = get_client()
    completion = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "What is 2 + 2?"}],
    )
    content = completion.choices[0].message.content
    finish = completion.choices[0].finish_reason
    print(f"  Content:       {content!r}")
    print(f"  Finish reason: {finish!r}")


def demo_system_prompt() -> None:
    """DEMO 2: Using a system message to set the assistant persona."""
    client = get_client()
    completion = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a pirate. Respond in pirate speak."},
            {"role": "user", "content": "What's the weather like?"},
        ],
    )
    print(f"  Response: {completion.choices[0].message.content!r}")


def demo_finish_reason_check() -> None:
    """DEMO 3: Proper finish_reason checking before using content."""
    client = get_client()
    completion = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "Write a haiku."}],
        max_tokens=50,
    )
    reason = completion.choices[0].finish_reason
    content = completion.choices[0].message.content

    if reason == "stop":
        print(f"  Complete response: {content!r}")
    elif reason == "length":
        print(f"  Truncated (hit max_tokens): {content!r}")
    elif reason == "content_filter":
        print("  Content was filtered by safety system")
    else:
        print(f"  Unexpected finish_reason: {reason!r}")


def anti_pattern_no_finish_check() -> None:
    """ANTI-PATTERN 1: Using content without checking finish_reason.

    If the response was cut off (reason="length"), the content is incomplete.
    Treating it as complete leads to silent data corruption.
    """
    client = get_client()
    completion = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "List 100 countries."}],
        max_tokens=20,
    )
    # WRONG: blindly using content without checking finish_reason
    # content = completion.choices[0].message.content
    # process(content)   # silently uses truncated text!

    # CORRECT:
    reason = completion.choices[0].finish_reason
    print(f"  finish_reason={reason!r} — must check before trusting content")


if __name__ == "__main__":
    sep = "=" * 60
    mode = "MOCK" if is_mock() else "LIVE"
    print(f"{NL}{sep}{NL}Domain 1 - Task 1.2: Chat Completions Basics [{mode}]{NL}{sep}")

    print(f"{NL}--- DEMO 1: Basic Chat ---")
    demo_basic_chat()

    print(f"{NL}--- DEMO 2: System Prompt ---")
    demo_system_prompt()

    print(f"{NL}--- DEMO 3: finish_reason Check ---")
    demo_finish_reason_check()

    print(f"{NL}--- ANTI-PATTERN 1: No Finish Check ---")
    anti_pattern_no_finish_check()

    print(f"{NL}KEY TAKEAWAYS:")
    print("  1. Access response text via choices[0].message.content")
    print("  2. ALWAYS check finish_reason: 'stop'=complete, 'length'=truncated")
    print("  3. System messages set persona; user messages drive the conversation")
    print("  4. max_tokens limits output length — finish_reason tells you if it hit the cap")
