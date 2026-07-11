"""Typed configuration loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class Config:
    """Application configuration with sensible defaults."""

    model: str = "gpt-4o"
    embedding_model: str = "text-embedding-3-small"
    max_tokens: int = 1024
    temperature: float = 0.7
    max_retries: int = 3
    timeout: float = 30.0
    api_key: str | None = None

    @classmethod
    def from_env(cls) -> Config:
        """Load configuration from environment variables."""
        return cls(
            model=os.environ.get("OPENAI_MODEL", "gpt-4o"),
            embedding_model=os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
            max_tokens=int(os.environ.get("OPENAI_MAX_TOKENS", "1024")),
            temperature=float(os.environ.get("OPENAI_TEMPERATURE", "0.7")),
            max_retries=int(os.environ.get("OPENAI_MAX_RETRIES", "3")),
            timeout=float(os.environ.get("OPENAI_TIMEOUT", "30.0")),
            api_key=os.environ.get("OPENAI_API_KEY"),
        )

    @property
    def is_mock(self) -> bool:
        """True when no API key is configured."""
        return not bool(self.api_key and self.api_key.strip())
