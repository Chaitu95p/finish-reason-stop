"""Domain 3 - Task 3.5: Few-Shot Examples for Structure

CONCEPTS:
  1. Few-shot examples — show the model desired input/output pairs
  2. Example as user+assistant message pairs before the real prompt
  3. Constrains format without needing JSON schema
  4. When to use — complex nested format, unusual schema, fine-tuning alternative

Mnemonic: FUSE — Few_shot, User_examples, Show_format, Enforce

Run:
  uv run python 05_few_shot_for_structure.py
"""

NL = chr(10)
MODEL = "gpt-4o"

from shared.mock import get_client, is_mock


def build_few_shot_messages(query: str) -> list[dict[str, str]]:
    """Build messages list with 2 few-shot examples before the real query."""
    return [
        {"role": "system", "content": "Extract structured data from text. Always output JSON."},
        # Example 1
        {"role": "user", "content": "Parse: 'Alice, 28, engineer from Seattle'"},
        {"role": "assistant", "content": '{"name":"Alice","age":28,"role":"engineer","city":"Seattle"}'},
        # Example 2
        {"role": "user", "content": "Parse: 'Bob, 35, designer from NYC'"},
        {"role": "assistant", "content": '{"name":"Bob","age":35,"role":"designer","city":"NYC"}'},
        # Real query
        {"role": "user", "content": f"Parse: '{query}'"},
    ]


def build_zero_shot_messages(query: str) -> list[dict[str, str]]:
    """Build messages list without examples (zero-shot)."""
    return [
        {"role": "system", "content": "Extract structured data from text. Output JSON."},
        {"role": "user", "content": f"Parse: '{query}'"},
    ]


def demo_few_shot_comparison() -> None:
    """DEMO 1: Compare few-shot vs zero-shot messages structure."""
    query = "Carol, 42, manager from Chicago"

    few_shot_msgs = build_few_shot_messages(query)
    zero_shot_msgs = build_zero_shot_messages(query)

    print(f"  Zero-shot message count: {len(zero_shot_msgs)}")
    print(f"  Few-shot message count:  {len(few_shot_msgs)}")
    print(f"{NL}  Few-shot messages (showing structure):")
    for i, msg in enumerate(few_shot_msgs):
        content_preview = msg["content"][:50] + ("..." if len(msg["content"]) > 50 else "")
        print(f"    [{i}] {msg['role']}: {content_preview!r}")


def demo_few_shot_request() -> None:
    """DEMO 2: Send few-shot messages to the API."""
    client = get_client()
    query = "David, 31, data scientist from Boston"
    messages = build_few_shot_messages(query)

    completion = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        response_format={"type": "json_object"},
    )
    result = completion.choices[0].message.content
    print(f"  Query: {query!r}")
    print(f"  Response: {result!r}")
    print(f"  finish_reason: {completion.choices[0].finish_reason!r}")


def demo_when_to_use_few_shot() -> None:
    """DEMO 3: Decision guide — few-shot vs JSON schema."""
    print("  When to use few-shot examples:")
    print("    ✓ Complex nested format hard to express as JSON Schema")
    print("    ✓ Domain-specific formatting conventions")
    print("    ✓ When you have real examples from production data")
    print("    ✓ Prototyping before writing full Pydantic models")
    print(f"{NL}  When to use JSON Schema instead:")
    print("    ✓ Strict type enforcement required")
    print("    ✓ Schema is stable and well-defined")
    print("    ✓ You need Pydantic validation downstream")
    print("    ✓ additionalProperties: false matters for security")


if __name__ == "__main__":
    sep = "=" * 60
    mode = "MOCK" if is_mock() else "LIVE"
    print(f"{NL}{sep}{NL}Domain 3 - Task 3.5: Few-Shot for Structure [{mode}]{NL}{sep}")

    print(f"{NL}--- DEMO 1: Message Structure Comparison ---")
    demo_few_shot_comparison()

    print(f"{NL}--- DEMO 2: Few-Shot API Request ---")
    demo_few_shot_request()

    print(f"{NL}--- DEMO 3: When to Use Each ---")
    demo_when_to_use_few_shot()

    print(f"{NL}KEY TAKEAWAYS:")
    print("  1. Few-shot examples teach format by demonstration, not specification")
    print("  2. Use user+assistant pairs in the messages list before the real query")
    print("  3. 2-3 examples often outperform 0-shot with complex schemas")
    print("  4. Combine few-shot + json_object mode for best results without Pydantic")
