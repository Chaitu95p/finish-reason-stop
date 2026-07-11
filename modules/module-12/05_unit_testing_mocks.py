"""Domain 12 - Task 12.5: Unit Testing with Mocks

CONCEPTS:
  1. unittest.mock.patch — swap openai.OpenAI with a mock in tests
  2. Test the logic, not the API — assert on what YOUR code does with the response
  3. MockClient from shared.mock — zero-config mock for unit tests
  4. parametrize — run the same test logic with multiple inputs

Mnemonic: PTMR — Patch_client, Test_logic, Mock_response, Run_parametrized

Run:
  uv run python 05_unit_testing_mocks.py
"""

NL = chr(10)
MODEL = "gpt-4o"

import unittest
from unittest.mock import MagicMock

from shared.mock import MockClient, is_mock

# --- The production code under test ---

def classify_sentiment(client: object, text: str) -> str:
    """Return 'positive', 'negative', or 'neutral' for text."""
    response = client.chat.completions.create(  # type: ignore[attr-defined]
        model=MODEL,
        messages=[
            {"role": "system", "content": "Reply with one word: positive, negative, or neutral."},
            {"role": "user", "content": text},
        ],
    )
    return (response.choices[0].message.content or "").strip().lower()


def summarize_text(client: object, text: str, max_words: int = 50) -> str:
    """Summarize text in at most max_words words."""
    prompt = f"Summarize in at most {max_words} words: {text}"
    response = client.chat.completions.create(  # type: ignore[attr-defined]
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content or ""


# --- Tests using MockClient from shared.mock ---

class TestWithMockClient(unittest.TestCase):
    """Tests using shared.mock.MockClient — no patch() needed."""

    def setUp(self) -> None:
        self.client = MockClient()

    def test_classify_sentiment_returns_string(self) -> None:
        result = classify_sentiment(self.client, "I love this!")
        self.assertIsInstance(result, str)

    def test_summarize_returns_content(self) -> None:
        result = summarize_text(self.client, "A long article about science and technology.")
        self.assertIsInstance(result, str)
        self.assertTrue(len(result) > 0)


# --- Tests using unittest.mock.patch for controlled behavior ---

class TestWithPatch(unittest.TestCase):
    """Tests with patch() to control exact response content."""

    def _make_mock_response(self, content: str) -> MagicMock:
        """Build a minimal mock chat completion."""
        mock_msg = MagicMock()
        mock_msg.content = content
        mock_choice = MagicMock()
        mock_choice.message = mock_msg
        mock_choice.finish_reason = "stop"
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        return mock_response

    def test_classify_positive(self) -> None:
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = self._make_mock_response("positive")
        result = classify_sentiment(mock_client, "This is great!")
        self.assertEqual(result, "positive")

    def test_classify_negative(self) -> None:
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = self._make_mock_response("negative")
        result = classify_sentiment(mock_client, "This is terrible!")
        self.assertEqual(result, "negative")

    def test_classify_handles_extra_whitespace(self) -> None:
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = self._make_mock_response("  POSITIVE  ")
        result = classify_sentiment(mock_client, "Amazing!")
        self.assertEqual(result, "positive")


def demo_run_tests() -> None:
    """DEMO 1: Run the test suite."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TestWithMockClient))
    suite.addTests(loader.loadTestsFromTestCase(TestWithPatch))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    print(f"{NL}  Tests run: {result.testsRun}")
    print(f"  Failures:  {len(result.failures)}")
    print(f"  Errors:    {len(result.errors)}")


def demo_testing_principles() -> None:
    """DEMO 2: What to test and what not to test."""
    print("  Test YOUR logic, not OpenAI's API:")
    print("    ✓ Does your code parse the response correctly?")
    print("    ✓ Does your code handle empty content?")
    print("    ✓ Does your code strip whitespace from model output?")
    print("    ✗ Don't test that OpenAI returns a valid response (that's their job)")
    print(f"{NL}  Test levels:")
    print("    Unit:        Mock the client — test parsing and business logic")
    print("    Integration: Real client with real API — smoke tests only")
    print("    Contract:    Validate response schema matches your Pydantic models")


if __name__ == "__main__":
    sep = "=" * 60
    mode = "MOCK" if is_mock() else "LIVE"
    print(f"{NL}{sep}{NL}Domain 12 - Task 12.5: Unit Testing [{mode}]{NL}{sep}")

    print(f"{NL}--- DEMO 1: Run Tests ---")
    demo_run_tests()

    print(f"{NL}--- DEMO 2: Testing Principles ---")
    demo_testing_principles()

    print(f"{NL}KEY TAKEAWAYS:")
    print("  1. Use MockClient from shared.mock for zero-config unit tests")
    print("  2. Use unittest.mock.MagicMock when you need exact response content")
    print("  3. Test YOUR code's behavior — parsing, validation, error handling")
    print("  4. Never test that the OpenAI API returns correct data — that's not your responsibility")
