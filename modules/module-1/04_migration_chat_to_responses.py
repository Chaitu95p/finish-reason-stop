"""Domain 1 - Task 1.4: Migrating from Chat Completions to Responses API

CONCEPTS:
  1. Equivalent patterns — messages/input, choices/output, system role/instructions
  2. System prompt migration — system role message → instructions param
  3. Tool use migration — same tool schemas, different loop termination logic
  4. previous_response_id replaces manually appended assistant messages

Mnemonic: MITRE — Migrate, Input, Tools, Responses, Equivalent

Run:
  uv run python 04_migration_chat_to_responses.py
"""

NL = chr(10)
MODEL = "gpt-4o"

from shared.mock import get_client, is_mock

# ---------- Before: Chat Completions approach ----------

def chat_completions_answer(question: str, system_prompt: str = "") -> str:
    """Answer a question using Chat Completions (legacy approach)."""
    client = get_client()
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": question})

    completion = client.chat.completions.create(model=MODEL, messages=messages)
    if completion.choices[0].finish_reason == "stop":
        return completion.choices[0].message.content or ""
    return ""


# ---------- After: Responses API approach ----------

def responses_api_answer(question: str, instructions: str = "") -> str:
    """Answer a question using the Responses API (recommended approach)."""
    client = get_client()
    kwargs: dict = {"model": MODEL, "input": question}
    if instructions:
        kwargs["instructions"] = instructions
    response = client.responses.create(**kwargs)
    return response.output_text


def show_migration_table() -> None:
    """DEMO: Side-by-side migration mapping."""
    print("  Migration Mapping:")
    mappings = [
        ("messages=[{'role':'system','content':X}]", "instructions=X"),
        ("messages=[{'role':'user','content':Q}]", "input=Q"),
        ("choices[0].message.content", "response.output_text"),
        ("choices[0].finish_reason == 'stop'", "response.status == 'completed'"),
        ("Append assistant msg + new user msg", "previous_response_id=response.id"),
        ("finish_reason == 'tool_calls'", "Check output item types for 'tool_call'"),
    ]
    print(f"  {'Chat Completions':<45} {'Responses API'}")
    print(f"  {'-'*45} {'-'*40}")
    for old, new in mappings:
        print(f"  {old:<45} {new}")


def demo_equivalent_calls() -> None:
    """DEMO: Run the same task through both APIs, confirm same result pattern."""
    q = "What color is the sky?"
    instructions = "Answer in exactly three words."

    chat_result = chat_completions_answer(q, system_prompt=instructions)
    resp_result = responses_api_answer(q, instructions=instructions)

    print(f"  Chat Completions result:  {chat_result!r}")
    print(f"  Responses API result:     {resp_result!r}")
    print(f"  (Both use same MODEL={MODEL!r} and instructions)")


if __name__ == "__main__":
    sep = "=" * 60
    mode = "MOCK" if is_mock() else "LIVE"
    print(f"{NL}{sep}{NL}Domain 1 - Task 1.4: Migrating Chat → Responses [{mode}]{NL}{sep}")

    print(f"{NL}--- DEMO 1: Migration Mapping ---")
    show_migration_table()

    print(f"{NL}--- DEMO 2: Equivalent Calls ---")
    demo_equivalent_calls()

    print(f"{NL}KEY TAKEAWAYS:")
    print("  1. system role → instructions param (Responses API)")
    print("  2. messages[user] → input param (string or list of content blocks)")
    print("  3. choices[0].message.content → response.output_text")
    print("  4. Manual message history → previous_response_id for stateful turns")
