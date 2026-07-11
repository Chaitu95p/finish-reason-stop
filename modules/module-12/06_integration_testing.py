"""Domain 12 - Task 12.6: Integration Testing

CONCEPTS:
  1. Smoke tests — minimal live API calls to verify end-to-end connectivity
  2. Contract tests — verify response schema matches your Pydantic models
  3. Environment gating — skip live tests when OPENAI_API_KEY is absent
  4. Recorded responses — VCR-style cassettes for reproducible CI tests

Mnemonic: SCER — Smoke, Contract, Env_gate, Recorded

Run:
  uv run python 06_integration_testing.py
"""

NL = chr(10)
MODEL = "gpt-4o-mini"

import os
import unittest
from typing import Any

from pydantic import BaseModel, ValidationError
from shared.mock import MockClient, is_mock


class ChatResponse(BaseModel):
    """Contract schema for a chat completion response."""
    content: str
    finish_reason: str
    prompt_tokens: int
    completion_tokens: int


def parse_chat_response(raw_response: Any) -> ChatResponse:
    """Parse raw API response into typed ChatResponse."""
    choice = raw_response.choices[0]
    usage = raw_response.usage
    return ChatResponse(
        content=choice.message.content or "",
        finish_reason=choice.finish_reason,
        prompt_tokens=getattr(usage, "prompt_tokens", 0),
        completion_tokens=getattr(usage, "completion_tokens", 0),
    )


# --- Smoke tests ---

class SmokeTests(unittest.TestCase):
    """Fast smoke tests — verify basic connectivity and response shape."""

    @unittest.skipUnless(os.environ.get("OPENAI_API_KEY"), "OPENAI_API_KEY not set")
    def test_chat_completions_live(self) -> None:
        """Verify the live API returns a valid response."""
        import openai
        client = openai.OpenAI()
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": "Say 'ok'"}],
            max_tokens=5,
        )
        self.assertEqual(response.choices[0].finish_reason, "stop")

    def test_chat_completions_mock(self) -> None:
        """Verify MockClient returns expected shape."""
        client = MockClient()
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": "Hello"}],
        )
        self.assertEqual(response.choices[0].finish_reason, "stop")
        self.assertIsInstance(response.choices[0].message.content, str)


# --- Contract tests ---

class ContractTests(unittest.TestCase):
    """Verify response shape matches our Pydantic contract schema."""

    def test_response_matches_contract(self) -> None:
        client = MockClient()
        raw = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": "Test"}],
        )
        try:
            parsed = parse_chat_response(raw)
            self.assertIsInstance(parsed.content, str)
            self.assertIn(parsed.finish_reason, {"stop", "tool_calls", "length", "content_filter"})
        except ValidationError as e:
            self.fail(f"Contract validation failed: {e}")


def demo_smoke_tests() -> None:
    """DEMO 1: Run smoke and contract tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(SmokeTests))
    suite.addTests(loader.loadTestsFromTestCase(ContractTests))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    print(f"{NL}  Tests run: {result.testsRun}")
    print(f"  Failures:  {len(result.failures)}")
    skipped = len(result.skipped)
    print(f"  Skipped:   {skipped} (live tests require OPENAI_API_KEY)")


def demo_vcr_pattern() -> None:
    """DEMO 2: VCR cassette pattern for reproducible CI tests."""
    print("  VCR cassette pattern — record once, replay forever:")
    print("""
    # Record: run once with live API, save response to cassette file
    # Replay: load cassette, return recorded response (no API call)

    import vcrpy  # pip install vcrpy

    @vcr.use_cassette('cassettes/test_summarize.yaml')
    def test_summarize_article():
        client = openai.OpenAI()
        # First run: calls live API, saves to cassette
        # Subsequent runs: uses cassette — no API key or network needed
        response = client.chat.completions.create(...)
        assert \"summary\" in response.choices[0].message.content.lower()
    """)
    print("  Benefits:")
    print("    ✓ Reproducible CI tests without API key")
    print("    ✓ No cost for repeated test runs")
    print("    ✓ Deterministic — same response every time")
    print("    Note: cassettes must be refreshed when prompt or expected format changes")


def demo_env_gating() -> None:
    """DEMO 3: Environment-gated test execution."""
    print("  Environment gating pattern:")
    print("""
    @unittest.skipUnless(
        os.environ.get(\"OPENAI_API_KEY\"),
        \"OPENAI_API_KEY not set — skipping live test\"
    )
    def test_live_api(self) -> None:
        ...
    """)
    print("  Or with pytest.mark.skipif:")
    print("""
    @pytest.mark.skipif(
        not os.environ.get(\"OPENAI_API_KEY\"),
        reason=\"OPENAI_API_KEY not set\"
    )
    def test_live_endpoint() -> None:
        ...
    """)
    print(f"  Current environment: OPENAI_API_KEY {'is set' if not is_mock() else 'is NOT set (mock mode)'}")


if __name__ == "__main__":
    sep = "=" * 60
    mode = "MOCK" if is_mock() else "LIVE"
    print(f"{NL}{sep}{NL}Domain 12 - Task 12.6: Integration Testing [{mode}]{NL}{sep}")

    print(f"{NL}--- DEMO 1: Smoke + Contract Tests ---")
    demo_smoke_tests()

    print(f"{NL}--- DEMO 2: VCR Cassette Pattern ---")
    demo_vcr_pattern()

    print(f"{NL}--- DEMO 3: Environment Gating ---")
    demo_env_gating()

    print(f"{NL}KEY TAKEAWAYS:")
    print("  1. Smoke tests: verify end-to-end shape with minimal live calls")
    print("  2. Contract tests: validate response schema against your Pydantic models")
    print("  3. Gate live tests on OPENAI_API_KEY — skipUnless pattern")
    print("  4. VCR cassettes let you run integration tests in CI without an API key")
