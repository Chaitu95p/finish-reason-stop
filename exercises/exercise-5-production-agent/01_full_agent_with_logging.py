"""Exercise 5 - Task 5.1: Production Agent with Structured Logging

GOAL: Build a production-quality agent that logs every API call
with request_id, model, tokens, cost, and latency.

SKILLS PRACTICED:
  - Structured JSON logging on every API call
  - Cost tracking per request and per session
  - Correlation IDs for request tracing

Run:
  uv run python 01_full_agent_with_logging.py
"""

NL = chr(10)
MODEL = "gpt-4o-mini"

import json
import time
from typing import Any

from shared.logging import new_request_id, structured_log
from shared.mock import get_client, is_mock
from shared.tokens import estimate_cost

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_fact",
            "description": "Get a fact about a topic.",
            "parameters": {
                "type": "object",
                "properties": {"topic": {"type": "string"}},
                "required": ["topic"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    },
]


def get_fact(topic: str) -> str:
    """Simulated fact retrieval tool."""
    facts = {
        "python": "Python was created by Guido van Rossum in 1991.",
        "openai": "OpenAI was founded in December 2015.",
        "default": f"Interesting fact about {topic}.",
    }
    return json.dumps({"fact": facts.get(topic.lower(), facts["default"])})


def logged_api_call(client: object, messages: list[dict], tools: list[dict]) -> Any:
    """Make an API call and emit a structured log entry."""
    request_id = new_request_id()
    t0 = time.monotonic()
    response = client.chat.completions.create(  # type: ignore[attr-defined]
        model=MODEL,
        messages=messages,
        tools=tools,
    )
    latency_ms = (time.monotonic() - t0) * 1000

    usage = response.usage
    prompt_tokens = getattr(usage, "prompt_tokens", 10)
    completion_tokens = getattr(usage, "completion_tokens", 5)
    cost = estimate_cost(prompt_tokens, completion_tokens, MODEL)

    structured_log(
        "api_call",
        request_id=request_id,
        model=MODEL,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        finish_reason=response.choices[0].finish_reason,
        latency_ms=round(latency_ms, 1),
        cost_usd=round(cost, 6),
    )
    return response


def run_logged_agent(user_message: str) -> str:
    """Run agent loop with structured logging on every API call."""
    client = get_client()
    messages: list[dict] = [{"role": "user", "content": user_message}]
    session_cost = 0.0

    while True:
        response = logged_api_call(client, messages, TOOLS)
        choice = response.choices[0]
        msg = choice.message
        tool_calls = getattr(msg, "tool_calls", None) or []

        assistant_msg: dict[str, Any] = {"role": "assistant", "content": msg.content}
        if tool_calls:
            assistant_msg["tool_calls"] = [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in tool_calls
            ]
        messages.append(assistant_msg)

        if choice.finish_reason == "stop":
            structured_log("session_complete", session_cost_usd=round(session_cost, 6))
            return msg.content or ""

        if choice.finish_reason == "tool_calls":
            for tc in tool_calls:
                args = json.loads(tc.function.arguments or "{}")
                result = get_fact(**args)
                structured_log("tool_call", tool=tc.function.name, args=args, result=result)
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

    return ""


if __name__ == "__main__":
    sep = "=" * 60
    mode = "MOCK" if is_mock() else "LIVE"
    print(f"{NL}{sep}{NL}Exercise 5 - Task 5.1: Logged Agent [{mode}]{NL}{sep}")

    result = run_logged_agent("Tell me a fact about Python and a fact about OpenAI.")
    print(f"{NL}  Final answer: {result!r}")

    print(f"{NL}KEY TAKEAWAYS:")
    print("  1. Log every API call: request_id, model, tokens, cost, latency, finish_reason")
    print("  2. Log tool calls separately: tool name, args, result")
    print("  3. Emit session_complete with total cost for billing tracking")
