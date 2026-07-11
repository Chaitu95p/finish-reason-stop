"""Domain 2 - Task 2.4: Parallel Tool Calls

CONCEPTS:
  1. Multiple tool calls in one response — model batches related lookups
  2. Execute ALL tool calls before the next API call
  3. Append ALL tool results together (one per tool_call_id)
  4. Order: execute in any order; append in tool_call_id order

Mnemonic: BEAT — Batch, Execute_all, Append_all, Together

Run:
  uv run python 04_parallel_tool_calls.py
"""

NL = chr(10)
MODEL = "gpt-4o"

import json
from dataclasses import dataclass

from shared.mock import (
    MockFunction,
    MockToolCall,
    is_mock,
)


@dataclass
class StockPrice:
    ticker: str
    price_usd: float
    change_pct: float


def get_stock_price(ticker: str) -> StockPrice:
    prices = {"AAPL": (182.50, 1.2), "GOOGL": (140.20, -0.5), "MSFT": (378.90, 0.8)}
    price, change = prices.get(ticker.upper(), (100.0, 0.0))
    return StockPrice(ticker=ticker.upper(), price_usd=price, change_pct=change)


def execute_tool_call(tool_call: MockToolCall) -> str:
    name = tool_call.function.name
    args: dict = json.loads(tool_call.function.arguments)
    if name == "get_stock_price":
        result = get_stock_price(args["ticker"])
        return json.dumps({"ticker": result.ticker, "price": result.price_usd, "change_pct": result.change_pct})
    return json.dumps({"error": f"Unknown tool: {name}"})


STOCK_TOOL = {
    "type": "function",
    "function": {
        "name": "get_stock_price",
        "description": "Get current stock price for a ticker symbol.",
        "parameters": {
            "type": "object",
            "properties": {"ticker": {"type": "string", "description": "Stock ticker e.g. AAPL"}},
            "required": ["ticker"],
            "additionalProperties": False,
        },
    },
}


def demo_parallel_tool_calls() -> None:
    """DEMO 1: Simulate model returning multiple tool calls simultaneously."""
    # In real API, model returns multiple tool_calls when it needs parallel data.
    # We simulate this since MockClient returns one at a time.
    simulated_tool_calls = [
        MockToolCall(id="call_001", function=MockFunction(name="get_stock_price", arguments='{"ticker":"AAPL"}')),
        MockToolCall(id="call_002", function=MockFunction(name="get_stock_price", arguments='{"ticker":"GOOGL"}')),
        MockToolCall(id="call_003", function=MockFunction(name="get_stock_price", arguments='{"ticker":"MSFT"}')),
    ]

    messages: list[dict] = [
        {"role": "user", "content": "Compare AAPL, GOOGL, and MSFT stock prices."}
    ]

    # Append simulated assistant message with all 3 tool calls
    messages.append({
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {"id": tc.id, "type": tc.type, "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
            for tc in simulated_tool_calls
        ],
    })

    print(f"  Executing {len(simulated_tool_calls)} tool calls in parallel...")
    # Execute ALL tool calls before calling API again
    for tc in simulated_tool_calls:
        result = execute_tool_call(tc)
        print(f"    {tc.id}: {tc.function.name}({tc.function.arguments}) → {result}")
        messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

    print(f"  Messages list now has {len(messages)} items")
    print("  Ready to call API again with all results appended")


def anti_pattern_partial_execution() -> None:
    """ANTI-PATTERN 1: Executing only the first tool call and ignoring the rest.

    If the model returned 3 tool calls and you only respond to 1,
    the API will return a 400 error: "Missing tool_call_id for ..."
    You MUST respond to every tool_call in a response.
    """
    simulated_calls = [
        MockToolCall(id="call_A", function=MockFunction(name="get_stock_price", arguments='{"ticker":"AAPL"}')),
        MockToolCall(id="call_B", function=MockFunction(name="get_stock_price", arguments='{"ticker":"GOOGL"}')),
    ]
    messages: list[dict] = [{"role": "user", "content": "Compare stocks"}]
    messages.append({
        "role": "assistant", "content": None,
        "tool_calls": [{"id": tc.id, "type": tc.type, "function": {"name": tc.function.name, "arguments": tc.function.arguments}} for tc in simulated_calls]
    })

    # WRONG: only respond to first tool call
    first_result = execute_tool_call(simulated_calls[0])
    messages.append({"role": "tool", "tool_call_id": simulated_calls[0].id, "content": first_result})
    print("  ANTI-PATTERN: only responded to call_A, ignored call_B")
    print("  → API would return 400: missing tool result for call_B")
    print("  FIX: loop over ALL tool_calls and append a tool message for each")


if __name__ == "__main__":
    sep = "=" * 60
    mode = "MOCK" if is_mock() else "LIVE"
    print(f"{NL}{sep}{NL}Domain 2 - Task 2.4: Parallel Tool Calls [{mode}]{NL}{sep}")

    print(f"{NL}--- DEMO 1: Execute All Tool Calls ---")
    demo_parallel_tool_calls()

    print(f"{NL}--- ANTI-PATTERN 1: Partial Execution ---")
    anti_pattern_partial_execution()

    print(f"{NL}KEY TAKEAWAYS:")
    print("  1. Model may return multiple tool_calls in one response — always handle all")
    print("  2. Execute all tool calls before calling the API again")
    print("  3. Append one tool result message per tool_call_id — order doesn't matter")
    print("  4. Missing a tool result causes a 400 API error in production")
