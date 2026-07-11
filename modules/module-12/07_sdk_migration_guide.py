"""Domain 12 - Task 12.7: SDK Migration Guide

CONCEPTS:
  1. v0 → v1 breaking changes — client construction, error imports, streaming
  2. Chat Completions → Responses API — mapping of concepts
  3. Deprecation warnings — how the SDK signals upcoming removals
  4. Codemod hints — sed/AST transforms for bulk migration

Mnemonic: VCDR — V0_v1_changes, Chat_to_responses, Deprecation_warnings, Replace_patterns

Run:
  uv run python 07_sdk_migration_guide.py
"""

NL = chr(10)
MODEL = "gpt-4o"

from shared.mock import get_client, is_mock

MIGRATION_V0_V1 = [
    ("import openai; openai.ChatCompletion.create(...)", "client.chat.completions.create(...)"),
    ("import openai; openai.Embedding.create(...)",      "client.embeddings.create(...)"),
    ("import openai; openai.Image.create(...)",          "client.images.generate(...)"),
    ("from openai.error import RateLimitError",         "from openai import RateLimitError"),
    ("from openai.error import OpenAIError",            "from openai import OpenAIError"),
    ("response['choices'][0]['message']['content']",    "response.choices[0].message.content"),
    ("openai.api_key = os.environ['OPENAI_API_KEY']",   "client = OpenAI()  # reads env automatically"),
]

CHAT_TO_RESPONSES = [
    ("client.chat.completions.create()",               "client.responses.create()"),
    ("messages=[{\"role\":\"system\",\"content\":...}]",    "instructions=\"...\""),
    ("messages=[{\"role\":\"user\",\"content\":...}]",      "input=\"...\""),
    ("choices[0].message.content",                      "response.output_text"),
    ("choices[0].finish_reason == 'stop'",              "response.status == 'completed'"),
    ("previous messages list",                          "previous_response_id"),
]


def demo_v0_to_v1() -> None:
    """DEMO 1: openai v0 → v1 breaking changes."""
    print("  v0 → v1 migration table:")
    print(f"  {'v0 (< 1.0.0)':<50} {'v1 (>= 1.0.0)'}")
    print(f"  {'-'*50} {'-'*40}")
    for old, new in MIGRATION_V0_V1:
        print(f"  {old:<50} {new}")


def demo_chat_to_responses() -> None:
    """DEMO 2: Chat Completions → Responses API migration."""
    print("  Chat Completions → Responses API concept mapping:")
    print(f"  {'Chat Completions':<45} {'Responses API'}")
    print(f"  {'-'*45} {'-'*35}")
    for chat, responses in CHAT_TO_RESPONSES:
        print(f"  {chat:<45} {responses}")


def demo_side_by_side_code() -> None:
    """DEMO 3: Side-by-side code comparison."""
    client = get_client()
    print("  Chat Completions (still works, but not primary):")
    chat_response = client.chat.completions.create(  # type: ignore[attr-defined]
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are helpful."},
            {"role": "user",   "content": "Name one planet."},
        ],
    )
    chat_content = chat_response.choices[0].message.content or ""
    print(f"    choices[0].message.content: {chat_content!r}")

    print(f"{NL}  Responses API (preferred):")
    resp_response = client.responses.create(  # type: ignore[attr-defined]
        model=MODEL,
        instructions="You are helpful.",
        input="Name one planet.",
    )
    print(f"    output_text: {resp_response.output_text!r}")
    print(f"    status:      {resp_response.status!r}")


def demo_codemod_hints() -> None:
    """DEMO 4: Sed/grep commands for bulk migration."""
    print("  Bulk migration search patterns (grep / sed):")
    patterns = [
        ("Find v0 module-level calls", "grep -r 'openai\\.ChatCompletion' src/"),
        ("Find v0 error imports",      "grep -r 'from openai.error import' src/"),
        ("Find dict-access responses", "grep -r \"\\['choices'\\]\" src/"),
        ("Find api_key assignment",    "grep -r 'openai\\.api_key' src/"),
    ]
    for desc, cmd in patterns:
        print(f"    # {desc}")
        print(f"    {cmd}")
        print()


def demo_version_check() -> None:
    """DEMO 5: Check current SDK version at runtime."""
    try:
        import openai
        version = openai.__version__
        major = int(version.split(".")[0])
        print(f"  openai SDK version: {version!r}")
        if major >= 1:
            print("  ✓ Using v1 SDK — current API style")
        else:
            print("  ✗ Using v0 SDK — migrate to v1")
    except Exception as e:
        print(f"  Could not check version: {e}")


if __name__ == "__main__":
    sep = "=" * 60
    mode = "MOCK" if is_mock() else "LIVE"
    print(f"{NL}{sep}{NL}Domain 12 - Task 12.7: SDK Migration Guide [{mode}]{NL}{sep}")

    print(f"{NL}--- DEMO 1: v0 → v1 Changes ---")
    demo_v0_to_v1()

    print(f"{NL}--- DEMO 2: Chat → Responses API ---")
    demo_chat_to_responses()

    print(f"{NL}--- DEMO 3: Side-by-Side Code ---")
    demo_side_by_side_code()

    print(f"{NL}--- DEMO 4: Codemod Hints ---")
    demo_codemod_hints()

    print(f"{NL}--- DEMO 5: Version Check ---")
    demo_version_check()

    print(f"{NL}KEY TAKEAWAYS:")
    print("  1. v1 SDK uses a client object — no more module-level openai.ChatCompletion.create()")
    print("  2. Errors import from openai directly — 'from openai import RateLimitError'")
    print("  3. Response fields are attributes, not dict keys — response.choices[0] not response['choices'][0]")
    print("  4. Responses API is the preferred modern API — Chat Completions still works but is secondary")
