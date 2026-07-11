"""Domain 1 - Task 1.3: Responses API vs Chat Completions

CONCEPTS:
  1. Responses API — stateful, output-centric, built for agents
  2. Chat Completions — messages-list, widely supported, widely documented
  3. Output access — response.output_text vs choices[0].message.content
  4. When to use each — Responses for new projects, Chat for compatibility

Mnemonic: SNOW — Stateful, New, Output, When

Run:
  uv run python 03_responses_vs_chat.py
"""

NL = chr(10)
MODEL = "gpt-4o"

from shared.mock import get_client, is_mock

PROMPT = "Explain photosynthesis in one sentence."


def responses_api_approach() -> str:
    """Use the Responses API — newer, stateful, output-centric."""
    client = get_client()
    response = client.responses.create(
        model=MODEL,
        input=PROMPT,
    )
    return response.output_text


def chat_completions_approach() -> str:
    """Use Chat Completions — classic, messages-list approach."""
    client = get_client()
    completion = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": PROMPT}],
    )
    return completion.choices[0].message.content or ""


def compare_response_structures() -> None:
    """DEMO: Show structural differences between both APIs."""
    client = get_client()

    # Responses API
    resp = client.responses.create(model=MODEL, input="Hi")
    print("  Responses API response fields:")
    print(f"    .output_text = {resp.output_text!r}")
    print(f"    .status      = {resp.status!r}")
    print(f"    .id          = {resp.id!r}")
    print(f"    .output      = list of {len(resp.output)} item(s)")

    # Chat Completions
    chat = client.chat.completions.create(
        model=MODEL, messages=[{"role": "user", "content": "Hi"}]
    )
    print(f"{NL}  Chat Completions response fields:")
    print(f"    .choices[0].message.content = {chat.choices[0].message.content!r}")
    print(f"    .choices[0].finish_reason   = {chat.choices[0].finish_reason!r}")
    print(f"    .id                         = {chat.id!r}")


def decision_matrix() -> None:
    """DEMO: When to choose each API."""
    print("  Decision Matrix:")
    rows = [
        ("Agentic/tool loops", "Responses API", "Chat Completions"),
        ("Stateful multi-turn", "Responses API", "Chat Completions (manual)"),
        ("Third-party SDK compat.", "Chat Completions", "Responses API"),
        ("Streaming", "Either", "Either"),
        ("Legacy codebases", "Chat Completions", "Responses API"),
        ("New greenfield projects", "Responses API", "Chat Completions"),
    ]
    print(f"  {'Use Case':<30} {'Prefer':<20} {'Avoid'}")
    print(f"  {'-'*30} {'-'*20} {'-'*20}")
    for use_case, prefer, avoid in rows:
        print(f"  {use_case:<30} {prefer:<20} {avoid}")


if __name__ == "__main__":
    sep = "=" * 60
    mode = "MOCK" if is_mock() else "LIVE"
    print(f"{NL}{sep}{NL}Domain 1 - Task 1.3: Responses API vs Chat Completions [{mode}]{NL}{sep}")

    print(f"{NL}--- DEMO 1: Same Prompt, Both APIs ---")
    r_text = responses_api_approach()
    c_text = chat_completions_approach()
    print(f"  Responses API: {r_text!r}")
    print(f"  Chat Completions: {c_text!r}")

    print(f"{NL}--- DEMO 2: Structural Differences ---")
    compare_response_structures()

    print(f"{NL}--- DEMO 3: Decision Matrix ---")
    decision_matrix()

    print(f"{NL}KEY TAKEAWAYS:")
    print("  1. Responses API: stateful (previous_response_id), newer, agent-first design")
    print("  2. Chat Completions: messages-list, required for most third-party integrations")
    print("  3. For new projects: prefer Responses API; for compatibility: Chat Completions")
    print("  4. Both support tools, streaming, structured output — choose by state model")
