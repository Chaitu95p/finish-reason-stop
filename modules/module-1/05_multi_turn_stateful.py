"""Domain 1 - Task 1.5: Multi-Turn Stateful Conversations

CONCEPTS:
  1. previous_response_id — chain responses; server manages history
  2. Manual messages list — client manages history (Chat Completions style)
  3. Statefulness tradeoffs — server-managed vs client-managed context
  4. Conversation reset — starting fresh by omitting previous_response_id

Mnemonic: PRAMS — Previous_response_id, Responses, And, Messages, Stateful

Run:
  uv run python 05_multi_turn_stateful.py
"""

NL = chr(10)
MODEL = "gpt-4o"

from shared.mock import get_client, is_mock


def demo_stateful_responses_api() -> None:
    """DEMO 1: Multi-turn via previous_response_id — no manual history."""
    client = get_client()

    # Turn 1
    r1 = client.responses.create(model=MODEL, input="My name is Alice.")
    print(f"  Turn 1 — ID: {r1.id!r}  Reply: {r1.output_text!r}")

    # Turn 2 — chain via previous_response_id
    r2 = client.responses.create(
        model=MODEL,
        input="What is my name?",
        previous_response_id=r1.id,
    )
    print(f"  Turn 2 — ID: {r2.id!r}  Reply: {r2.output_text!r}")

    # Turn 3 — continue the chain
    r3 = client.responses.create(
        model=MODEL,
        input="Say my name in reverse.",
        previous_response_id=r2.id,
    )
    print(f"  Turn 3 — ID: {r3.id!r}  Reply: {r3.output_text!r}")
    print("  (No local history management needed!)")


def demo_manual_history_chat() -> None:
    """DEMO 2: Multi-turn via manual messages list (Chat Completions)."""
    client = get_client()
    messages: list[dict[str, str]] = []

    # Turn 1
    messages.append({"role": "user", "content": "My name is Bob."})
    c1 = client.chat.completions.create(model=MODEL, messages=messages)
    assistant_reply = c1.choices[0].message.content or ""
    messages.append({"role": "assistant", "content": assistant_reply})
    print(f"  Turn 1 — Reply: {assistant_reply!r}")

    # Turn 2 — append next user message
    messages.append({"role": "user", "content": "What is my name?"})
    c2 = client.chat.completions.create(model=MODEL, messages=messages)
    reply2 = c2.choices[0].message.content or ""
    messages.append({"role": "assistant", "content": reply2})
    print(f"  Turn 2 — Reply: {reply2!r}")
    print(f"  (Messages list length: {len(messages)} — grows each turn)")


def anti_pattern_infinite_chain() -> None:
    """ANTI-PATTERN 1: Endless previous_response_id chain without reset.

    Chaining indefinitely accumulates context tokens without pruning.
    This raises costs and eventually hits context limits.
    """
    client = get_client()
    prev_id: str | None = None

    # WRONG: chaining 100+ turns with no reset or summarization strategy
    # In real use, you'd periodically summarize and reset:
    for turn in range(3):  # shortened for demo; real bug is unbounded loop
        response = client.responses.create(
            model=MODEL,
            input=f"Turn {turn + 1} message.",
            **({"previous_response_id": prev_id} if prev_id else {}),
        )
        prev_id = response.id

    print(f"  Last chain ID: {prev_id!r}")
    print("  ANTI-PATTERN: no pruning/summarization strategy defined")
    print("  FIX: summarize context every N turns and start fresh chain")


if __name__ == "__main__":
    sep = "=" * 60
    mode = "MOCK" if is_mock() else "LIVE"
    print(f"{NL}{sep}{NL}Domain 1 - Task 1.5: Multi-Turn Stateful Conversations [{mode}]{NL}{sep}")

    print(f"{NL}--- DEMO 1: Responses API with previous_response_id ---")
    demo_stateful_responses_api()

    print(f"{NL}--- DEMO 2: Chat Completions Manual History ---")
    demo_manual_history_chat()

    print(f"{NL}--- ANTI-PATTERN 1: Infinite Chain ---")
    anti_pattern_infinite_chain()

    print(f"{NL}KEY TAKEAWAYS:")
    print("  1. previous_response_id: server manages history — no client-side list needed")
    print("  2. Chat Completions: you own the messages list — append each turn manually")
    print("  3. Both approaches accumulate context — plan a reset/summarize strategy")
    print("  4. To start fresh: omit previous_response_id (Responses) or clear messages (Chat)")
