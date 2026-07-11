"""Shared TypedDicts and Pydantic models used across modules."""

from __future__ import annotations

from typing import Literal, TypedDict

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# TypedDicts for message construction
# ---------------------------------------------------------------------------


class MessageDict(TypedDict):
    role: str
    content: str


class ToolResultDict(TypedDict):
    role: Literal["tool"]
    tool_call_id: str
    content: str


# ---------------------------------------------------------------------------
# Pydantic models for structured output demos
# ---------------------------------------------------------------------------


class SentimentResult(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"]
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str


class SummaryResult(BaseModel):
    title: str
    summary: str
    key_points: list[str]
    word_count: int = 0


class EntityExtraction(BaseModel):
    entities: list[str]
    entity_type: str
    context: str


class StepByStepReasoning(BaseModel):
    steps: list[str]
    conclusion: str
    confidence: float = Field(ge=0.0, le=1.0)


class SearchResult(BaseModel):
    query: str
    results: list[str]
    relevance_scores: list[float]
