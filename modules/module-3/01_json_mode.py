"""Domain 3 - Task 3.1: JSON Mode

CONCEPTS:
  1. response_format={"type":"json_object"} — forces JSON output
  2. json.loads() with try/except — always parse safely
  3. Must instruct model in prompt — JSON mode doesn't infer schema
  4. Limitations — no schema enforcement; model can produce any valid JSON

Mnemonic: JPOS — JSON_mode, Parse, Outline_in_prompt, Schema_less

Run:
  uv run python 01_json_mode.py
"""

NL = chr(10)
MODEL = "gpt-4o"

import json

from shared.mock import get_client, is_mock


def demo_json_mode() -> None:
    """DEMO 1: Request structured data via JSON mode."""
    client = get_client()
    completion = client.chat.completions.create(
        model=MODEL,
        messages=[{
            "role": "user",
            "content": (
                "Analyze this text and return JSON with keys: "
                "sentiment (positive/negative/neutral), confidence (0.0-1.0), "
                "key_phrases (array of strings). "
                "Text: 'The product was amazing and I loved every feature!'"
            ),
        }],
        response_format={"type": "json_object"},
    )
    raw = completion.choices[0].message.content or "{}"
    try:
        data = json.loads(raw)
        print(f"  Parsed JSON: {data}")
        print(f"  sentiment:   {data.get('sentiment', 'missing')!r}")
        print(f"  confidence:  {data.get('confidence', 'missing')}")
    except json.JSONDecodeError as e:
        print(f"  Parse error: {e}")


def demo_json_array() -> None:
    """DEMO 2: Request a JSON array of items."""
    client = get_client()
    completion = client.chat.completions.create(
        model=MODEL,
        messages=[{
            "role": "user",
            "content": "Return a JSON array of 3 programming language names as strings.",
        }],
        response_format={"type": "json_object"},
    )
    raw = completion.choices[0].message.content or "{}"
    try:
        data = json.loads(raw)
        print(f"  Result: {data}")
    except json.JSONDecodeError as e:
        print(f"  Parse error: {e}")


def anti_pattern_no_schema_in_prompt() -> None:
    """ANTI-PATTERN 1: JSON mode without telling model what structure to use.

    json_object forces valid JSON, but the model picks the structure.
    Without guidance, you get unpredictable shapes.
    """
    client = get_client()
    # WRONG: No schema specified in prompt
    completion = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "Tell me about Python. Return JSON."}],
        response_format={"type": "json_object"},
    )
    print("  ANTI-PATTERN: no schema in prompt → model invents structure")
    print(f"  Result shape: {type(completion.choices[0].message.content)}")
    print("  FIX: specify exact keys in the prompt")


def anti_pattern_no_try_except() -> None:
    """ANTI-PATTERN 2: json.loads() without error handling.

    Even in JSON mode, edge cases (empty content, malformed) can occur.
    """
    raw = ""
    # WRONG:
    # data = json.loads(raw)  # raises json.JSONDecodeError on empty string!

    # CORRECT:
    try:
        data = json.loads(raw or "{}")
        print(f"  Safe parse result: {data}")
    except json.JSONDecodeError as e:
        print(f"  Caught expected error: {e}")


if __name__ == "__main__":
    sep = "=" * 60
    mode = "MOCK" if is_mock() else "LIVE"
    print(f"{NL}{sep}{NL}Domain 3 - Task 3.1: JSON Mode [{mode}]{NL}{sep}")

    print(f"{NL}--- DEMO 1: JSON Mode Sentiment ---")
    demo_json_mode()

    print(f"{NL}--- DEMO 2: JSON Array ---")
    demo_json_array()

    print(f"{NL}--- ANTI-PATTERN 1: No Schema in Prompt ---")
    anti_pattern_no_schema_in_prompt()

    print(f"{NL}--- ANTI-PATTERN 2: No try/except ---")
    anti_pattern_no_try_except()

    print(f"{NL}KEY TAKEAWAYS:")
    print("  1. JSON mode forces valid JSON but NOT a specific schema — always describe it in prompt")
    print("  2. Always wrap json.loads() in try/except — even JSON mode can produce edge cases")
    print("  3. For schema enforcement, use json_schema with strict:true (see Task 3.2)")
    print("  4. JSON mode works with both Chat Completions and Responses API")
