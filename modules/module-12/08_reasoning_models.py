"""Domain 12 - Task 12.8: Reasoning Models

CONCEPTS:
  1. reasoning_effort — "low" / "medium" / "high" controls compute budget
  2. max_completion_tokens — replaces max_tokens for reasoning models
  3. reasoning_tokens — usage.completion_tokens_details.reasoning_tokens
  4. System message restriction — o1 models ignore system messages
  5. When to use reasoning models vs gpt-4o (latency/cost tradeoff)

Mnemonic: REMST — Reasoning_effort, Effort, Max_completion, System, Tradeoff

Run:
  uv run python 08_reasoning_models.py
"""

from __future__ import annotations

from shared.mock import get_client, is_mock

NL = chr(10)
MODEL = "o3"
MODEL_O1 = "o1"


def demo_basic_o3_call() -> None:
    """DEMO 1: Basic o3 call with reasoning_effort and max_completion_tokens."""
    client = get_client()
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "What is 2 + 2? Think step by step."}],
        reasoning_effort="medium",
        max_completion_tokens=1024,
    )
    choice = response.choices[0]
    print(f"Model          : {response.model}")
    print(f"finish_reason  : {choice.finish_reason}")
    print(f"Response       : {choice.message.content}")
    print(f"Total tokens   : {response.usage.total_tokens}")


def demo_reasoning_tokens() -> None:
    """DEMO 2: Reading reasoning_tokens from the usage breakdown."""
    client = get_client()
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "Prove that sqrt(2) is irrational."}],
        reasoning_effort="high",
        max_completion_tokens=2048,
    )
    usage = response.usage
    reasoning = usage.completion_tokens_details.reasoning_tokens
    completion = usage.completion_tokens
    print(f"completion_tokens          : {completion}")
    print(f"  reasoning_tokens         : {reasoning}  (internal chain-of-thought)")
    print(f"  visible_tokens           : {completion - reasoning}  (what you see in the response)")
    print(f"prompt_tokens              : {usage.prompt_tokens}")
    print(f"total_tokens               : {usage.total_tokens}")
    print()
    print(
        f"  Note: reasoning_tokens are billed at output price but never{NL}"
        f"  appear in the response text — they represent internal thinking."
    )


def demo_o1_no_system_message() -> None:
    """DEMO 3: Correct way to pass instructions to o1 — embed in user message."""
    client = get_client()

    # Correct: embed instructions in user message
    instructions = "You are a careful senior engineer. Be concise and precise."
    user_query = "Explain the difference between a mutex and a semaphore."
    combined_message = f"{instructions}{NL}{NL}{user_query}"

    response = client.chat.completions.create(
        model=MODEL_O1,
        messages=[{"role": "user", "content": combined_message}],
        max_completion_tokens=512,
    )
    print("Instructions embedded in user message (correct for o1):")
    print(f"  message sent : {combined_message[:80]!r}...")
    print(f"  response     : {response.choices[0].message.content}")


def anti_pattern_max_tokens() -> None:
    """ANTI-PATTERN 1: Using max_tokens with a reasoning model.

    o1/o3/o4 models accept max_completion_tokens, not max_tokens.
    Passing max_tokens is silently ignored — you get an uncapped response
    or an unexpected error depending on the SDK version.
    """
    client = get_client()
    print("ANTI-PATTERN: Using max_tokens with reasoning model")
    print("  Wrong : client.chat.completions.create(model='o3', max_tokens=100, ...)")
    print("  Right : client.chat.completions.create(model='o3', max_completion_tokens=100, ...)")
    print()

    # In mock mode we demonstrate the correct form:
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "Count to 3."}],
        max_completion_tokens=100,  # correct parameter
    )
    print(f"  Correct call succeeded. finish_reason={response.choices[0].finish_reason}")


def anti_pattern_system_message_o1() -> None:
    """ANTI-PATTERN 2: Relying on a system message with an o1 model.

    o1 models do not support the system role in the messages array.
    The message is silently dropped, causing unexpected behaviour.
    Use the developer role or embed the instructions in the user message.
    """
    print("ANTI-PATTERN: System message with o1 model")
    print("  Wrong:")
    print("    messages=[")
    print('      {"role": "system", "content": "You are a senior engineer."},')
    print('      {"role": "user", "content": "Explain mutexes."},')
    print("    ]")
    print("  -> System message is silently ignored by o1.")
    print()
    print("  Right:")
    print("    messages=[")
    print('      {"role": "user", "content": "You are a senior engineer.\\nExplain mutexes."},')
    print("    ]")
    print("  Or use the developer role (supported in some o1 variants):")
    print("    messages=[")
    print('      {"role": "developer", "content": "You are a senior engineer."},')
    print('      {"role": "user", "content": "Explain mutexes."},')
    print("    ]")


if __name__ == "__main__":
    sep = "=" * 60
    mode = "MOCK" if is_mock() else "LIVE"
    print(f"{NL}{sep}{NL}Domain 12 - Task 12.8: Reasoning Models [{mode}]{NL}{sep}")

    print(f"{NL}--- DEMO 1: Basic o3 Call ---")
    demo_basic_o3_call()

    print(f"{NL}--- DEMO 2: Reading reasoning_tokens ---")
    demo_reasoning_tokens()

    print(f"{NL}--- DEMO 3: Instructions for o1 (no system message) ---")
    demo_o1_no_system_message()

    print(f"{NL}--- ANTI-PATTERN 1: max_tokens vs max_completion_tokens ---")
    anti_pattern_max_tokens()

    print(f"{NL}--- ANTI-PATTERN 2: System message with o1 ---")
    anti_pattern_system_message_o1()

    print(f"{NL}KEY TAKEAWAYS:")
    print("  1. Use max_completion_tokens (not max_tokens) for o1/o3/o4 models")
    print("  2. reasoning_tokens in completion_tokens_details = internal chain-of-thought cost")
    print("  3. o1 ignores system messages — embed instructions in user or developer role")
    print("  4. reasoning_effort='low' for speed, 'high' for hard multi-step problems")
    print("  5. Use gpt-4o for latency-sensitive tasks; o3/o4 for complex reasoning")
