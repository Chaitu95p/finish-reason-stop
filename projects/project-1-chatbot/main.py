"""Project 1: Persistent CLI Chatbot with Tool Use

A production-quality CLI chatbot with:
- Persistent conversation history (up to MAX_HISTORY turns)
- Tool use: get_time and calculator tools
- Structured logging on every API call
- Mock mode: runs without OPENAI_API_KEY

Run:
  uv run python main.py
  uv run python main.py --demo  # non-interactive demo mode
"""

NL = chr(10)
MODEL = "gpt-4o-mini"
MAX_HISTORY = 10

import argparse
import json
import time
from typing import Any

from shared.logging import new_request_id, structured_log
from shared.mock import get_client, is_mock
from shared.tokens import estimate_cost, format_cost

SYSTEM_PROMPT = (
    "You are a helpful assistant with access to tools. "
    "Use the calculator tool for math and get_time for current time information."
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate an arithmetic expression.",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_time",
            "description": "Get the current UTC time as a string.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            },
            "strict": True,
        },
    },
]


def calculator(expression: str) -> str:
    try:
        allowed = set("0123456789+-*/()., ")
        if not all(c in allowed for c in expression):
            return json.dumps({"error": "invalid characters in expression"})
        result = eval(expression, {"__builtins__": {}})  # noqa: S307
        return json.dumps({"result": result})
    except Exception as e:
        return json.dumps({"error": str(e)})


def get_time() -> str:
    import datetime
    return json.dumps({"utc_time": datetime.datetime.utcnow().isoformat() + "Z"})


def execute_tool(name: str, arguments_json: str) -> str:
    try:
        args = json.loads(arguments_json or "{}")
    except json.JSONDecodeError:
        return json.dumps({"error": "invalid arguments"})
    if name == "calculator":
        return calculator(args.get("expression", ""))
    if name == "get_time":
        return get_time()
    return json.dumps({"error": f"unknown tool: {name}"})


class Chatbot:
    def __init__(self) -> None:
        self.client = get_client()
        self.messages: list[dict[str, Any]] = []
        self.session_cost = 0.0

    def _trim_history(self) -> None:
        if len(self.messages) > MAX_HISTORY * 2:
            self.messages = self.messages[-(MAX_HISTORY * 2):]

    def _call_api(self) -> Any:
        request_id = new_request_id()
        t0 = time.monotonic()
        full_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + self.messages
        response = self.client.chat.completions.create(  # type: ignore[attr-defined]
            model=MODEL,
            messages=full_messages,
            tools=TOOLS,
        )
        latency_ms = (time.monotonic() - t0) * 1000
        usage = response.usage
        pt = getattr(usage, "prompt_tokens", 10)
        ct = getattr(usage, "completion_tokens", 5)
        cost = estimate_cost(pt, ct, MODEL)
        self.session_cost += cost
        structured_log("api_call", request_id=request_id, model=MODEL,
                       prompt_tokens=pt, completion_tokens=ct,
                       latency_ms=round(latency_ms, 1), cost_usd=round(cost, 6))
        return response

    def chat(self, user_input: str) -> str:
        self.messages.append({"role": "user", "content": user_input})
        self._trim_history()

        for _turn in range(10):
            response = self._call_api()
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
            self.messages.append(assistant_entry)

            if choice.finish_reason == "stop":
                return msg.content or ""

            if choice.finish_reason == "tool_calls":
                for tc in tool_calls:
                    result = execute_tool(tc.function.name, tc.function.arguments)
                    self.messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

        return "Max turns reached."


def demo_mode(bot: Chatbot) -> None:
    """Run a scripted demo without user input."""
    demo_inputs = [
        "What is 1234 * 5678?",
        "What time is it right now?",
        "What was my first question?",
    ]
    for user_input in demo_inputs:
        print(f"You: {user_input}")
        response = bot.chat(user_input)
        print(f"Bot: {response}{NL}")


def interactive_mode(bot: Chatbot) -> None:
    """Run interactive CLI chat loop."""
    mode = "MOCK" if is_mock() else "LIVE"
    print(f"Chatbot [{mode}] | Model: {MODEL} | Type 'quit' to exit{NL}")
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"{NL}Goodbye!")
            break
        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print(f"Session cost: {format_cost(bot.session_cost)}")
            break
        response = bot.chat(user_input)
        print(f"Bot: {response}{NL}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CLI Chatbot with tool use")
    parser.add_argument("--demo", action="store_true", help="Run scripted demo")
    args = parser.parse_args()

    bot = Chatbot()
    if args.demo:
        demo_mode(bot)
    else:
        interactive_mode(bot)
