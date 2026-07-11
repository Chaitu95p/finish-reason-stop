"""Project 6: Research Agent — Multi-Tool Web Search + Synthesis

An agentic researcher that:
- Uses search, fetch_page, and take_notes tools in a proper loop
- Synthesizes findings into a structured research report
- Works in mock mode (simulated tool results)

Run:
  uv run python main.py
  uv run python main.py --demo
  uv run python main.py --query "History of the internet"
"""

NL = chr(10)
MODEL = "gpt-4o"
MAX_TURNS = 15

import argparse
import json
from typing import Any

from shared.mock import get_client, is_mock

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for information on a topic.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                },
                "required": ["query"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_page",
            "description": "Fetch the content of a web page by URL.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "URL to fetch"},
                },
                "required": ["url"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    },
    {
        "type": "function",
        "function": {
            "name": "take_note",
            "description": "Save an important finding to the research notes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "note": {"type": "string", "description": "The finding to record"},
                    "source": {"type": "string", "description": "URL or source name"},
                },
                "required": ["note", "source"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    },
]

SYSTEM_PROMPT = (
    "You are a research agent. Given a research question, use the available tools to:\n"
    "1. Search for relevant information\n"
    "2. Fetch key pages for details\n"
    "3. Take notes on important findings\n"
    "4. When you have enough information (3+ notes), synthesize a concise research report.\n"
    "Be thorough but efficient. Use take_note to record each key finding."
)


# --- Simulated tool implementations ---

_notes: list[dict[str, str]] = []


def web_search(query: str) -> str:
    """Simulated web search returning mock results."""
    results = [
        {"title": f"Overview of {query}", "url": f"https://example.com/{query.replace(' ', '-')}", "snippet": f"Comprehensive overview of {query} covering history, current state, and future directions."},
        {"title": f"{query} — Wikipedia", "url": f"https://en.wikipedia.org/wiki/{query.replace(' ', '_')}", "snippet": f"Wikipedia article on {query} with detailed background and references."},
        {"title": f"Recent developments in {query}", "url": f"https://news.example.com/{query.replace(' ', '-')}", "snippet": f"Latest news and research on {query} from the past year."},
    ]
    return json.dumps({"results": results})


def fetch_page(url: str) -> str:
    """Simulated page fetch returning mock content."""
    return json.dumps({
        "url": url,
        "title": f"Page: {url}",
        "content": f"This page contains detailed information from {url}. Key facts include historical context, current applications, and future prospects. Multiple perspectives are covered with citations.",
    })


def take_note(note: str, source: str) -> str:
    """Record a research finding."""
    _notes.append({"note": note, "source": source})
    return json.dumps({"status": "noted", "total_notes": len(_notes)})


def execute_tool(name: str, arguments_json: str) -> str:
    """Dispatch to tool function; never raises."""
    try:
        args = json.loads(arguments_json or "{}")
    except json.JSONDecodeError:
        return json.dumps({"error": "invalid arguments JSON"})
    try:
        if name == "web_search":
            return web_search(args["query"])
        if name == "fetch_page":
            return fetch_page(args["url"])
        if name == "take_note":
            return take_note(args["note"], args["source"])
        return json.dumps({"error": f"unknown tool: {name}"})
    except Exception as e:
        return json.dumps({"error": f"{type(e).__name__}: {e}"})


def run_research_agent(query: str) -> str:
    """Run the research agent loop and return the final report."""
    global _notes
    _notes = []

    client = get_client()
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": f"Research question: {query}"},
    ]

    for turn in range(MAX_TURNS):
        response = client.chat.completions.create(  # type: ignore[attr-defined]
            model=MODEL,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}] + messages,
            tools=TOOLS,
        )
        choice = response.choices[0]
        msg = choice.message
        tool_calls = getattr(msg, "tool_calls", None) or []

        assistant_entry: dict[str, Any] = {"role": "assistant", "content": msg.content}
        if tool_calls:
            assistant_entry["tool_calls"] = [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in tool_calls
            ]
        messages.append(assistant_entry)

        if choice.finish_reason == "stop":
            return msg.content or ""

        if choice.finish_reason == "tool_calls":
            for tc in tool_calls:
                result = execute_tool(tc.function.name, tc.function.arguments)
                args_parsed = json.loads(tc.function.arguments or "{}")
                print(f"  [tool:{tc.function.name}] {list(args_parsed.values())[0] if args_parsed else ''}...")
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

    return f"Research complete. Notes collected: {len(_notes)}"


def format_notes() -> str:
    if not _notes:
        return "No notes recorded."
    lines = ["Research notes:"]
    for i, n in enumerate(_notes, 1):
        lines.append(f"  [{i}] {n['note']} (source: {n['source']})")
    return NL.join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Research Agent")
    parser.add_argument("--demo", action="store_true", help="Run with sample query")
    parser.add_argument("--query", type=str, help="Research question")
    args = parser.parse_args()

    mode = "MOCK" if is_mock() else "LIVE"
    print(f"Research Agent [{mode}]{NL}")

    if args.query:
        query = args.query
    elif args.demo:
        query = "What are the key concepts in retrieval augmented generation (RAG)?"
    else:
        try:
            query = input("Research question: ").strip()
        except (EOFError, KeyboardInterrupt):
            query = "What are the key concepts in retrieval augmented generation (RAG)?"

    print(f"Query: {query!r}{NL}")
    report = run_research_agent(query)

    print(f"{NL}--- Research Notes ---")
    print(format_notes())
    print(f"{NL}--- Final Report ---")
    print(report)
