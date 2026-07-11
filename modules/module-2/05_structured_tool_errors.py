"""Domain 2 - Task 2.5: Returning Errors from Tools

CONCEPTS:
  1. Tools can return error strings — model reads and recovers
  2. Structured error format — {"error": "msg", "code": "CODE"}
  3. Model adapts — retries different args, or tells user gracefully
  4. Never crash on bad tool input — catch exceptions, return error JSON

Mnemonic: RICE — Return_error, Inform_model, Catch_all, Error_json

Run:
  uv run python 05_structured_tool_errors.py
"""

NL = chr(10)
MODEL = "gpt-4o"

import json
from dataclasses import dataclass

from shared.mock import is_mock


@dataclass
class CityNotFoundError(Exception):
    city: str


def get_weather_strict(city: str) -> dict:
    """Mock weather that fails on unknown cities."""
    known_cities = {"london", "paris", "tokyo", "new york"}
    if city.lower() not in known_cities:
        raise CityNotFoundError(city=city)
    return {"city": city, "temp_c": 20, "condition": "clear"}


def safe_get_weather(city: str) -> str:
    """Tool wrapper: always returns JSON, never raises."""
    try:
        result = get_weather_strict(city)
        return json.dumps(result)
    except CityNotFoundError as e:
        return json.dumps({
            "error": f"City not found: {e.city!r}",
            "code": "CITY_NOT_FOUND",
            "suggestion": "Try a major city like 'London' or 'Tokyo'",
        })
    except Exception as e:
        return json.dumps({"error": str(e), "code": "INTERNAL_ERROR"})


WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get weather for a city. Returns error JSON if city not found.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
            "additionalProperties": False,
        },
    },
}


def demo_error_recovery() -> None:
    """DEMO 1: Tool returns error; model reads it and adapts."""
    # Simulate unknown city
    error_result = safe_get_weather("Atlantis")
    print("  Tool called with 'Atlantis':")
    print(f"  Result: {error_result}")
    parsed = json.loads(error_result)
    print(f"  error code: {parsed.get('code')!r}")
    print(f"  suggestion: {parsed.get('suggestion')!r}")
    print("  → Model would retry with 'London' or inform the user")

    # Simulate known city
    success_result = safe_get_weather("Paris")
    print(f"{NL}  Tool called with 'Paris':")
    print(f"  Result: {success_result}")


def anti_pattern_raise_exception() -> None:
    """ANTI-PATTERN 1: Letting exceptions propagate out of tool functions.

    If a tool raises an exception, your agentic loop crashes.
    The API never receives the tool result, and the conversation is lost.
    Always catch and return structured JSON errors.
    """
    def bad_get_weather(city: str) -> str:
        if city == "Atlantis":
            raise ValueError(f"City not found: {city}")  # WRONG: this crashes the loop!
        return json.dumps({"city": city, "temp_c": 20})

    try:
        bad_get_weather("Atlantis")
    except ValueError as e:
        print(f"  ANTI-PATTERN: exception escaped tool function: {e}")
        print("  FIX: wrap in try/except, return json.dumps({'error': str(e)})")


if __name__ == "__main__":
    sep = "=" * 60
    mode = "MOCK" if is_mock() else "LIVE"
    print(f"{NL}{sep}{NL}Domain 2 - Task 2.5: Structured Tool Errors [{mode}]{NL}{sep}")

    print(f"{NL}--- DEMO 1: Error Recovery ---")
    demo_error_recovery()

    print(f"{NL}--- ANTI-PATTERN 1: Bare Exception ---")
    anti_pattern_raise_exception()

    print(f"{NL}KEY TAKEAWAYS:")
    print("  1. Tools should NEVER raise exceptions — always return JSON strings")
    print("  2. Include 'error', 'code', and 'suggestion' fields for model recovery")
    print("  3. Model reads error content and can retry with corrected arguments")
    print("  4. A crashed tool breaks the entire agentic loop — defensive coding is essential")
