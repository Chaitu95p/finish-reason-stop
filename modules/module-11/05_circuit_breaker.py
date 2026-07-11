"""Domain 11 - Task 11.5: Circuit Breaker Pattern

CONCEPTS:
  1. Circuit states — CLOSED (normal), OPEN (failing), HALF_OPEN (testing)
  2. Failure threshold — how many failures before opening the circuit
  3. Recovery timeout — seconds before testing if the service recovered
  4. Fast fail — in OPEN state, raise immediately without calling the API

Mnemonic: COHT — Closed, Open, Half_open, Threshold

Run:
  uv run python 05_circuit_breaker.py
"""

NL = chr(10)
MODEL = "gpt-4o"

import time
from dataclasses import dataclass, field
from enum import Enum

import openai
from shared.mock import get_client, is_mock


class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class CircuitBreaker:
    failure_threshold: int = 3
    recovery_timeout: float = 30.0

    _state: CircuitState = field(default=CircuitState.CLOSED, init=False)
    _failures: int = field(default=0, init=False)
    _opened_at: float = field(default=0.0, init=False)

    @property
    def state(self) -> CircuitState:
        if self._state == CircuitState.OPEN:
            if time.monotonic() - self._opened_at >= self.recovery_timeout:
                self._state = CircuitState.HALF_OPEN
        return self._state

    def call(self, fn: object, *args: object, **kwargs: object) -> object:
        """Execute fn; manage circuit state around the call."""
        state = self.state
        if state == CircuitState.OPEN:
            raise RuntimeError(f"Circuit OPEN — fast failing (opened {time.monotonic() - self._opened_at:.0f}s ago)")

        try:
            result = fn(*args, **kwargs)  # type: ignore[operator]
            self._on_success()
            return result
        except (openai.APIConnectionError, openai.RateLimitError, openai.InternalServerError):
            self._on_failure()
            raise

    def _on_success(self) -> None:
        self._failures = 0
        self._state = CircuitState.CLOSED

    def _on_failure(self) -> None:
        self._failures += 1
        if self._failures >= self.failure_threshold:
            self._state = CircuitState.OPEN
            self._opened_at = time.monotonic()


def demo_circuit_breaker_states() -> None:
    """DEMO 1: Circuit breaker state transitions."""
    cb = CircuitBreaker(failure_threshold=3, recovery_timeout=30.0)
    print(f"  Initial state: {cb.state.value}")
    print(f"  Failures: {cb._failures}")

    # Simulate failures to open the circuit
    for i in range(3):
        cb._on_failure()
        print(f"  After failure {i+1}: state={cb.state.value} failures={cb._failures}")

    # Simulate recovery timeout
    cb._opened_at = time.monotonic() - 31.0  # pretend 31 seconds have passed
    print(f"  After recovery timeout: state={cb.state.value}")

    # Success closes it
    cb._on_success()
    print(f"  After success: state={cb.state.value}")


def demo_fast_fail() -> None:
    """DEMO 2: Fast fail when circuit is OPEN."""
    cb = CircuitBreaker(failure_threshold=1, recovery_timeout=60.0)
    cb._on_failure()  # open the circuit
    print(f"  Circuit state: {cb.state.value}")
    try:
        cb.call(lambda: None)
    except RuntimeError as e:
        print(f"  Fast fail raised: {e!r}")
    print("  Circuit saved an API call — no network request made")


def demo_circuit_with_mock_calls() -> None:
    """DEMO 3: Circuit breaker wrapping real API calls."""
    client = get_client()
    cb = CircuitBreaker(failure_threshold=3, recovery_timeout=30.0)

    def make_chat_call() -> str:
        resp = client.chat.completions.create(  # type: ignore[attr-defined]
            model=MODEL,
            messages=[{"role": "user", "content": "Hello"}],
        )
        return resp.choices[0].message.content or ""

    for i in range(3):
        try:
            result = cb.call(make_chat_call)
            print(f"  Call {i+1}: SUCCESS state={cb.state.value} result={str(result)[:30]!r}")
        except RuntimeError as e:
            print(f"  Call {i+1}: FAST FAIL — {e}")
        except Exception as e:
            print(f"  Call {i+1}: API error — {type(e).__name__}")


def demo_design_notes() -> None:
    """DEMO 4: When and how to use circuit breakers."""
    print("  When to use a circuit breaker:")
    print("    ✓ Downstream service that can go down (not just 429)")
    print("    ✓ High-volume systems where cascading failures are a risk")
    print("    ✓ When you have a fallback (degraded mode, cached response)")
    print(f"{NL}  Not needed if:")
    print("    ✗ You only make a few API calls per minute")
    print("    ✗ OpenAI's built-in retry (max_retries=3) is sufficient")
    print(f"{NL}  Production libraries: use 'pybreaker' or 'tenacity' with circuit_breaker")


if __name__ == "__main__":
    sep = "=" * 60
    mode = "MOCK" if is_mock() else "LIVE"
    print(f"{NL}{sep}{NL}Domain 11 - Task 11.5: Circuit Breaker [{mode}]{NL}{sep}")

    print(f"{NL}--- DEMO 1: State Transitions ---")
    demo_circuit_breaker_states()

    print(f"{NL}--- DEMO 2: Fast Fail ---")
    demo_fast_fail()

    print(f"{NL}--- DEMO 3: Wrapping API Calls ---")
    demo_circuit_with_mock_calls()

    print(f"{NL}--- DEMO 4: Design Notes ---")
    demo_design_notes()

    print(f"{NL}KEY TAKEAWAYS:")
    print("  1. States: CLOSED (normal) → OPEN (failing fast) → HALF_OPEN (probing)")
    print("  2. OPEN state short-circuits calls immediately — no network round-trip")
    print("  3. recovery_timeout controls when HALF_OPEN testing begins")
    print("  4. Always have a fallback path for when the circuit is open")
