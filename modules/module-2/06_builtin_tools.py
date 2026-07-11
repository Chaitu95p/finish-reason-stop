"""Domain 2 - Task 2.6: Built-in Tools

CONCEPTS:
  1. web_search_preview — model searches the web automatically
  2. file_search — searches a vector store of uploaded files
  3. code_interpreter — model writes and executes Python code
  4. Schema differences — built-in tools use {"type": "tool_name"} not "function"

Mnemonic: WFC — Web_search, File_search, Code_interpreter

Run:
  uv run python 06_builtin_tools.py
"""

NL = chr(10)
MODEL = "gpt-4o"

from shared.mock import get_client, is_mock

WEB_SEARCH_TOOL = {"type": "web_search_preview"}

FILE_SEARCH_TOOL = {
    "type": "file_search",
    "vector_store_ids": ["vs_example_001"],
}

CODE_INTERPRETER_TOOL = {"type": "code_interpreter"}


def demo_builtin_tool_schemas() -> None:
    """DEMO 1: Show built-in tool schema structure vs custom function tools."""
    import json

    print("  Built-in tool schemas (no 'function' wrapper):")
    print(f"    web_search:       {json.dumps(WEB_SEARCH_TOOL)}")
    print(f"    file_search:      {json.dumps(FILE_SEARCH_TOOL)}")
    print(f"    code_interpreter: {json.dumps(CODE_INTERPRETER_TOOL)}")

    print(f"{NL}  Custom function tool schema (requires 'function' wrapper):")
    custom = {
        "type": "function",
        "function": {"name": "my_func", "description": "...", "parameters": {"type": "object", "properties": {}, "required": []}},
    }
    print(f"    {json.dumps(custom)[:80]}...")


def demo_responses_api_with_web_search() -> None:
    """DEMO 2: Responses API call with web_search_preview enabled."""
    client = get_client()
    response = client.responses.create(
        model=MODEL,
        input="What is the latest news about Python?",
        tools=[WEB_SEARCH_TOOL],
    )
    print(f"  Status: {response.status!r}")
    print(f"  Output: {response.output_text!r}")
    print("  (In live mode, model would search the web and cite sources)")


def demo_file_search_with_vector_store() -> None:
    """DEMO 3: Responses API call with file_search tool and vector store."""
    client = get_client()
    response = client.responses.create(
        model=MODEL,
        input="What does our documentation say about authentication?",
        tools=[{
            "type": "file_search",
            "vector_store_ids": ["vs_example_001"],
            "max_num_results": 5,
        }],
    )
    print(f"  Status: {response.status!r}")
    print(f"  Output: {response.output_text!r}")
    print("  (In live mode, model would search uploaded documents)")


def anti_pattern_custom_schema_for_builtin() -> None:
    """ANTI-PATTERN 1: Using function schema for built-in tools.

    Built-in tools like web_search_preview don't have a 'function' wrapper.
    Using the wrong schema causes a 400 API error.
    """
    wrong_web_search = {
        "type": "function",
        "function": {
            "name": "web_search_preview",  # WRONG: built-ins aren't functions
            "description": "Search the web",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    }
    print("  ANTI-PATTERN: using 'function' type for built-in tool")
    print(f"    {wrong_web_search}")
    print(f"{NL}  CORRECT: use just the type string")
    print(f"    {WEB_SEARCH_TOOL}")


if __name__ == "__main__":
    sep = "=" * 60
    mode = "MOCK" if is_mock() else "LIVE"
    print(f"{NL}{sep}{NL}Domain 2 - Task 2.6: Built-in Tools [{mode}]{NL}{sep}")

    print(f"{NL}--- DEMO 1: Tool Schema Structure ---")
    demo_builtin_tool_schemas()

    print(f"{NL}--- DEMO 2: web_search_preview ---")
    demo_responses_api_with_web_search()

    print(f"{NL}--- DEMO 3: file_search ---")
    demo_file_search_with_vector_store()

    print(f"{NL}--- ANTI-PATTERN 1: Wrong Schema Type ---")
    anti_pattern_custom_schema_for_builtin()

    print(f"{NL}KEY TAKEAWAYS:")
    print("  1. Built-in tools use {\"type\": \"tool_name\"} — no 'function' wrapper")
    print("  2. web_search_preview: model handles searching; you just enable the tool")
    print("  3. file_search requires vector_store_ids — configure in tool schema")
    print("  4. Mix built-in and custom tools freely in the same tools list")
