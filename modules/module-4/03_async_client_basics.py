"""Domain 4 - Task 4.3: Async Client Basics

CONCEPTS:
  1. AsyncOpenAI — async version of OpenAI client
  2. await client.chat.completions.create() — async API call
  3. asyncio.run(main()) — entry point pattern
  4. get_async_client() from shared.mock

Mnemonic: AAER — AsyncOpenAI, Await, Event_loop, Run

Run:
  uv run python 03_async_client_basics.py
"""

from __future__ import annotations

import asyncio

from shared.mock import get_async_client, is_mock

NL = chr(10)
MODEL = "gpt-4o"


async def ask_question(question: str) -> str:
    """Send a single async chat completion request and return the response text."""
    client = await get_async_client()
    response = await client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": question}],
    )
    content: str | None = response.choices[0].message.content
    return content or ""


async def show_async_model_info() -> None:
    """Demonstrate async client by printing model and response info."""
    client = await get_async_client()
    response = await client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a concise assistant."},
            {"role": "user", "content": "Name three benefits of async programming."},
        ],
    )
    print(f"Model used : {response.model}")
    print(f"Finish reason : {response.choices[0].finish_reason}")
    print(f"Response  : {response.choices[0].message.content}")


async def demo_async_streaming() -> str:
    """
    Demonstrate async streaming.

    Note: AsyncMockClient.chat.completions.create(stream=True) returns a
    _MockStreamContextManager which uses synchronous __enter__/__iter__.
    With the real AsyncOpenAI the stream would be async-iterable; with the
    mock we use the sync context manager form. The pattern is otherwise identical.
    """
    client = await get_async_client()
    # Await the create() call — returns the stream context manager
    stream = await client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "Count to five briefly."}],
        stream=True,
    )
    accumulated: list[str] = []
    # Mock returns sync context manager — use 'with', not 'async with'
    with stream:
        for chunk in stream:
            content: str | None = chunk.choices[0].delta.content
            if content is not None:
                accumulated.append(content)
    return "".join(accumulated)


# ANTI-PATTERN 1: calling asyncio.run() inside an async function
async def anti_pattern_nested_run() -> None:
    """
    Wrong: calling asyncio.run() from within an async function.

    asyncio.run() creates a new event loop. Calling it inside a running
    event loop raises RuntimeError: 'This event loop is already running.'
    """
    print("ANTI-PATTERN: asyncio.run() inside async function")
    coro = ask_question("Will this work?")
    try:
        # This raises RuntimeError because we're already inside a running loop.
        asyncio.run(coro)
    except RuntimeError as exc:
        # Close the unawaited coroutine to suppress the ResourceWarning.
        coro.close()
        print(f"  RuntimeError caught (expected): {exc}")
    print(
        f"  Fix: await the coroutine directly — 'await ask_question(...)'{NL}"
        f"  Only call asyncio.run() from synchronous code (e.g. __main__ block)."
    )


# ANTI-PATTERN 2: forgetting to await get_async_client()
async def anti_pattern_unawaited_client() -> None:
    """
    Wrong: using get_async_client() without await.

    get_async_client is an async def — not awaiting it returns a coroutine
    object, not a client. Subsequent attribute access will raise AttributeError.
    """
    print("ANTI-PATTERN: forgetting to await get_async_client()")
    try:
        client_coro = get_async_client()  # Returns coroutine, NOT the client
        # Accessing .chat on a coroutine raises AttributeError
        _ = client_coro.chat  # type: ignore[union-attr]
    except AttributeError as exc:
        print(f"  AttributeError caught (expected): {exc}")
    finally:
        # Close the unawaited coroutine to avoid ResourceWarning
        if hasattr(client_coro, "close"):
            client_coro.close()  # type: ignore[union-attr]
    print("  Fix: client = await get_async_client()")


async def main() -> None:
    """Entry point for all async demos."""
    mock_label = "[MOCK]" if is_mock() else "[LIVE]"
    sep = "=" * 60

    print(f"Domain 4 - Task 4.3: Async Client Basics {mock_label}")
    print(sep)

    # DEMO 1: Basic async completion
    print("DEMO 1: Basic async chat completion")
    print(sep)
    answer = await ask_question("What does async/await mean in Python?")
    print("Question: 'What does async/await mean in Python?'")
    print(f"Answer  : {answer}")

    print(sep)

    # DEMO 2: Full response metadata
    print("DEMO 2: Async completion with response metadata")
    print(sep)
    await show_async_model_info()

    print(sep)

    # DEMO 3: Async streaming
    print("DEMO 3: Async streaming (sync context manager on mock)")
    print(sep)
    streamed = await demo_async_streaming()
    print(f"Streamed text: {streamed!r}")
    print(
        "Note: AsyncMockClient stream uses sync __iter__. "
        "Real AsyncOpenAI would use 'async for chunk in stream:'."
    )

    print(sep)

    # ANTI-PATTERN 1
    await anti_pattern_nested_run()

    print(sep)

    # ANTI-PATTERN 2
    await anti_pattern_unawaited_client()

    print(sep)
    print("KEY TAKEAWAYS:")
    print("  - Use get_async_client() (async def) — always await it to get the client")
    print("  - Async API calls use 'await client.chat.completions.create(...)' ")
    print("  - asyncio.run(main()) is the ONLY correct entry point; never nest it")
    print("  - Async clients allow concurrent I/O without threads — efficient for many requests")
    print("  - The mock async client uses asyncio.sleep(0) — functionally immediate")


if __name__ == "__main__":
    asyncio.run(main())
