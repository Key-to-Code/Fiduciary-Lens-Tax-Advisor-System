"""
Pydantic schemas for the public tax Q&A chat endpoint.
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


_ALLOWED_PROVIDERS = {"openai", "ollama", "local", "extractive", "auto"}


class ChatTurn(BaseModel):
    question: str = Field(..., min_length=1, max_length=4000)
    answer: str = Field(..., min_length=1, max_length=50_000)


class AskRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        max_length=4000,
        description="Natural-language question about Indian tax law.",
    )
    provider: Optional[str] = Field(
        None,
        description="Optional LLM provider override.",
    )
    history: List[ChatTurn] = Field(
        default_factory=list,
        description="Recent question/answer pairs for conversational context (not a legal source).",
        max_length=6,
    )

    @field_validator("question")
    @classmethod
    def strip_question(cls, value: str) -> str:
        cleaned = (value or "").strip()
        if not cleaned:
            raise ValueError("Question cannot be empty.")
        return cleaned

    @field_validator("provider")
    @classmethod
    def validate_provider(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        cleaned = value.strip().lower()
        if cleaned not in _ALLOWED_PROVIDERS:
            allowed = ", ".join(sorted(_ALLOWED_PROVIDERS))
            raise ValueError(f"Invalid provider '{value}'. Allowed: {allowed}")
        return cleaned


class SourceItem(BaseModel):
    n: int
    citation: str
    short: str
    document: str
    as_of: Optional[str] = None
    chunk_id: Optional[str] = None
    score: Optional[float] = None
    cosine: Optional[float] = None
    excerpt: str = ""
    url: Optional[str] = None


class AskResponse(BaseModel):
    success: bool = True
    answer: str
    sources: List[SourceItem] = Field(default_factory=list)
    grounded: bool
    provider: str
    latency_ms: int
    message: str = "Answer generated from the tax knowledge base."


class KnowledgeStatusResponse(BaseModel):
    available: bool
    ready: bool | None = None
    n_chunks: int = 0
    documents: List[str] = Field(default_factory=list)
    embed_model: Optional[str] = None
    label: str
