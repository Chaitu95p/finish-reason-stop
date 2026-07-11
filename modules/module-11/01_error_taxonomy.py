"""Domain 11 - Task 11.1: Error Taxonomy

CONCEPTS:
  1. openai.APIError — base class for all SDK errors
  2. APIConnectionError — network/DNS failures; always safe to retry
  3. RateLimitError — 429; use exponential backoff
  4. APIStatusError — non-retryable (400 bad request, 401 auth, 404 not found)
  5. APITimeoutError — request exceeded timeout; safe to retry

Mnemonic: CRSTA — Connection, Rate, Status, Timeout, All_from_APIError

Run:
  uv run python 01_error_taxonomy.py
"""

NL = chr(10)
MODEL = "gpt-4o"


from shared.mock import is_mock

ERROR_TAXONOMY = {
    "openai.APIConnectionError": {
        "cause": "Network failure, DNS error, server unreachable",
        "status": "N/A",
        "retryable": True,
        "action": "Retry with exponential backoff",
    },
    "openai.RateLimitError": {
        "cause": "Too many requests (429)",
        "status": 429,
        "retryable": True,
        "action": "Exponential backoff; check Retry-After header",
    },
    "openai.AuthenticationError": {
        "cause": "Invalid or missing API key (401)",
        "status": 401,
        "retryable": False,
        "action": "Check OPENAI_API_KEY env var; do not retry",
    },
    "openai.PermissionDeniedError": {
        "cause": "API key lacks access to this resource (403)",
        "status": 403,
        "retryable": False,
        "action": "Upgrade plan or request access; do not retry",
    },
    "openai.NotFoundError": {
        "cause": "Resource does not exist (404)",
        "status": 404,
        "retryable": False,
        "action": "Check model name or resource ID; do not retry",
    },
    "openai.BadRequestError": {
        "cause": "Invalid parameters (400)",
        "status": 400,
        "retryable": False,
        "action": "Fix request parameters; do not retry",
    },
    "openai.APITimeoutError": {
        "cause": "Request exceeded timeout setting",
        "status": "N/A",
        "retryable": True,
        "action": "Retry or increase OpenAI(timeout=...) setting",
    },
    "openai.InternalServerError": {
        "cause": "Server-side error (500/502/503/504)",
        "status": "500-504",
        "retryable": True,
        "action": "Retry with backoff; transient server issues",
    },
}


def demo_error_hierarchy() -> None:
    """DEMO 1: Print the openai exception hierarchy."""
    print("  openai exception hierarchy:")
    print("    openai.OpenAIError")
    print("      └─ openai.APIError")
    print("           ├─ openai.APIConnectionError")
    print("           │    └─ openai.APITimeoutError")
    print("           ├─ openai.APIStatusError")
    print("           │    ├─ openai.BadRequestError (400)")
    print("           │    ├─ openai.AuthenticationError (401)")
    print("           │    ├─ openai.PermissionDeniedError (403)")
    print("           │    ├─ openai.NotFoundError (404)")
    print("           │    ├─ openai.ConflictError (409)")
    print("           │    ├─ openai.UnprocessableEntityError (422)")
    print("           │    ├─ openai.RateLimitError (429)")
    print("           │    └─ openai.InternalServerError (500)")
    print("           └─ openai.APIResponseValidationError")


def demo_taxonomy_table() -> None:
    """DEMO 2: Full error taxonomy with retryability."""
    print(f"  {'Exception':<40} {'Status':<8} {'Retry?'}")
    print(f"  {'-'*40} {'-'*8} {'-'*6}")
    for exc, info in ERROR_TAXONOMY.items():
        retry_str = "YES" if info["retryable"] else "no"
        print(f"  {exc:<40} {str(info['status']):<8} {retry_str}")


def demo_catch_patterns() -> None:
    """DEMO 3: Correct catch pattern hierarchy."""
    print("  CORRECT: catch specific exceptions before the base class")
    print("""
    try:
        response = client.chat.completions.create(...)
    except openai.RateLimitError:
        # 429 — retry with backoff
    except openai.APIConnectionError:
        # network error — retry
    except openai.AuthenticationError:
        # bad API key — do NOT retry
    except openai.BadRequestError as e:
        # fix parameters — inspect e.body for details
    except openai.APIStatusError as e:
        # catch-all for other HTTP errors
        if e.status_code in {500, 502, 503, 504}:
            pass  # retry
    """)

    print("  ANTI-PATTERN 1: bare except hides error details")
    print("    try: ... except: pass  ← never do this")
    print(f"{NL}  ANTI-PATTERN 2: catching APIError before RateLimitError")
    print("    except openai.APIError: pass  ← catches everything, can't distinguish 429 from 400")


def demo_status_code_check() -> None:
    """DEMO 4: Reading status code and request ID from APIStatusError."""
    print("  Reading error details from APIStatusError:")
    print("""
    except openai.APIStatusError as e:
        print(e.status_code)   # e.g. 429
        print(e.response)      # httpx.Response object
        print(e.body)          # dict: {"error": {"message": ..., "type": ..., "code": ...}}
        print(e.request_id)    # X-Request-ID for support tickets
    """)


if __name__ == "__main__":
    sep = "=" * 60
    mode = "MOCK" if is_mock() else "LIVE"
    print(f"{NL}{sep}{NL}Domain 11 - Task 11.1: Error Taxonomy [{mode}]{NL}{sep}")

    print(f"{NL}--- DEMO 1: Exception Hierarchy ---")
    demo_error_hierarchy()

    print(f"{NL}--- DEMO 2: Taxonomy Table ---")
    demo_taxonomy_table()

    print(f"{NL}--- DEMO 3: Catch Patterns ---")
    demo_catch_patterns()

    print(f"{NL}--- DEMO 4: Reading Error Details ---")
    demo_status_code_check()

    print(f"{NL}KEY TAKEAWAYS:")
    print("  1. All openai errors inherit from openai.APIError (not Exception directly)")
    print("  2. Retryable: APIConnectionError, RateLimitError, APITimeoutError, 5xx")
    print("  3. NOT retryable: 400 (bad request), 401 (auth), 403 (permission), 404 (not found)")
    print("  4. Always catch specific exceptions before the base class APIStatusError")
