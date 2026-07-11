"""Project 4: Natural Language to SQL Generator

Converts natural language questions to SQL queries with:
- Schema-aware generation (schema injected into system prompt)
- Validation retry loop (re-prompts on syntax errors)
- Read-only enforcement (rejects SELECT alternatives, blocks mutations)

Run:
  uv run python main.py
  uv run python main.py --demo
"""

NL = chr(10)
MODEL = "gpt-4o"
MAX_RETRIES = 3

import argparse
import re

from pydantic import BaseModel
from shared.mock import get_client, is_mock

DB_SCHEMA = """
Tables:
  users(id INT, name VARCHAR, email VARCHAR, created_at TIMESTAMP)
  orders(id INT, user_id INT, product_id INT, amount DECIMAL, status VARCHAR, created_at TIMESTAMP)
  products(id INT, name VARCHAR, price DECIMAL, category VARCHAR, stock INT)

Relationships:
  orders.user_id → users.id
  orders.product_id → products.id
"""

SYSTEM_PROMPT = f"""You are a SQL expert. Generate SQLite SELECT queries from natural language.

Database schema:
{DB_SCHEMA}

Rules:
1. Generate only SELECT queries — never INSERT, UPDATE, DELETE, DROP, or ALTER
2. Use proper JOIN syntax for related tables
3. Return ONLY the SQL query — no explanation, no markdown, no code blocks
"""


class SQLResult(BaseModel):
    sql: str
    is_valid: bool
    error: str | None = None


MUTATION_PATTERN = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|TRUNCATE|REPLACE)\b",
    re.IGNORECASE,
)


def validate_sql(sql: str) -> tuple[bool, str | None]:
    """Basic SQL validation: must be SELECT and no mutation keywords."""
    stripped = sql.strip().upper()
    if not stripped.startswith("SELECT"):
        return False, "Query must start with SELECT"
    if MUTATION_PATTERN.search(sql):
        return False, "Mutation keywords detected (INSERT/UPDATE/DELETE/etc.)"
    if len(sql) < 7:
        return False, "Query too short"
    return True, None


def generate_sql(client: object, question: str) -> SQLResult:
    """Generate SQL with validation retry loop."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    last_error: str | None = None

    for attempt in range(MAX_RETRIES):
        if last_error and attempt > 0:
            messages.append({"role": "assistant", "content": "(previous attempt was invalid)"})
            messages.append({"role": "user", "content": f"The previous query was invalid: {last_error}. Please fix it."})

        response = client.chat.completions.create(  # type: ignore[attr-defined]
            model=MODEL,
            messages=messages,
        )
        sql = (response.choices[0].message.content or "").strip()
        sql = re.sub(r"^```sql\s*", "", sql, flags=re.IGNORECASE)
        sql = re.sub(r"\s*```$", "", sql)
        sql = sql.strip()

        is_valid, error = validate_sql(sql)
        if is_valid:
            return SQLResult(sql=sql, is_valid=True)

        last_error = error
        print(f"  [retry {attempt+1}] Invalid SQL: {error}")

    return SQLResult(sql=sql, is_valid=False, error=last_error)


def demo_mode(client: object) -> None:
    questions = [
        "How many users do we have?",
        "Show me the top 5 products by revenue",
        "Find users who placed orders in the last 30 days",
        "What is the average order amount by product category?",
    ]
    for q in questions:
        print(f"Q: {q}")
        result = generate_sql(client, q)
        if result.is_valid:
            print(f"SQL: {result.sql}")
        else:
            print(f"ERROR: {result.error}")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NL to SQL Generator")
    parser.add_argument("--demo", action="store_true", help="Run with sample questions")
    args = parser.parse_args()

    mode = "MOCK" if is_mock() else "LIVE"
    print(f"SQL Generator [{mode}]{NL}")

    client = get_client()
    if args.demo:
        demo_mode(client)
    else:
        print("Enter natural language questions (empty line to quit):")
        while True:
            try:
                q = input("Q: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not q:
                break
            result = generate_sql(client, q)
            if result.is_valid:
                print(f"SQL: {result.sql}{NL}")
            else:
                print(f"Error: {result.error}{NL}")
