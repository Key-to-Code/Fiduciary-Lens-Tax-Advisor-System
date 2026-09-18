"""
RAG/NLP service integration layer.

Stage 3: Integrates the FastAPI backend with the EXISTING RAG and NLP pipelines:
  - rag_model/src/generation/answer.py  →  TaxQA
  - nlp_pipeline/src/summarize.py       →  summarize_document
  - nlp_pipeline/src/insight.py         →  generate_insights
  - nlp_pipeline/src/extract.py         →  extract_financial_entities

The existing pipelines are NEVER duplicated, rewritten, or bypassed.
Imports are deferred to keep server startup fast and resilient.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any


class ProviderError(Exception):
    """Raised when an underlying LLM or retrieval backend fails."""

    def __init__(self, title: str, detail: str | None = None):
        super().__init__(title, detail)
        self.title = title
        self.detail = detail


# ---------------------------------------------------------------------------
# Legal Document Summarization (wraps summarize_document from nlp_pipeline)
# ---------------------------------------------------------------------------

def summarize_legal_document(
    text: str,
    provider: str | None = None,
) -> dict[str, Any]:
    """
    Summarize legal document text using the existing NLP pipeline.

    Parameters
    ----------
    text:
        The validated text extracted from the uploaded legal document.
    provider:
        Optional LLM provider name ("openai", "ollama", "local", "extractive", "auto").

    Returns
    -------
    dict with keys:
        summary: str
            The generated legal document summary.
        processing_time: float
            Execution time in seconds.

    Raises
    ------
    ProviderError
        If the configured LLM provider fails (e.g. connection error, missing key).
    """
    started = time.perf_counter()

    # Deferred import of the existing summarization function
    from nlp_pipeline.src.summarize import summarize_document  # noqa: PLC0415
    from shared import config  # noqa: PLC0415

    # If an explicit provider is passed, set it temporarily for the call
    original_provider = config.LLM_PROVIDER
    if provider:
        config.LLM_PROVIDER = provider

    try:
        summary = summarize_document(text)
    except Exception as exc:
        raise ProviderError(
            "LLM provider failure",
            f"The language model provider failed to generate a summary: {type(exc).__name__}: {str(exc)}",
        ) from exc
    finally:
        if provider:
            config.LLM_PROVIDER = original_provider

    processing_time = round(time.perf_counter() - started, 2)
    return {
        "summary": summary,
        "processing_time": processing_time,
    }


# ---------------------------------------------------------------------------
# RAG Q&A (wraps TaxQA from rag_model)
# ---------------------------------------------------------------------------

def ask_tax_question(question: str, provider: str | None = None) -> dict[str, Any]:
    """
    Ask a single tax question against the Income-tax Act 2025 knowledge base.

    Parameters
    ----------
    question:
        The natural-language tax question.
    provider:
        Optional provider override.

    Returns
    -------
    dict with answer text, sources, grounded flag, provider, latency_ms.
    """
    from rag_model.src.generation.answer import TaxQA  # noqa: PLC0415

    try:
        qa = TaxQA(provider=provider)
        answer = qa.ask(question)
    except Exception as exc:
        raise ProviderError(
            "RAG pipeline failure",
            f"Failed to query knowledge base: {type(exc).__name__}: {str(exc)}",
        ) from exc

    return {
        "answer": answer.text,
        "sources": answer.sources,
        "grounded": answer.grounded,
        "provider": answer.provider,
        "latency_ms": answer.latency_ms,
    }


# ---------------------------------------------------------------------------
# Document Insight Pipeline (wraps generate_insights from nlp_pipeline)
# ---------------------------------------------------------------------------

def analyse_document(file_path: str | Path) -> dict[str, Any]:
    """
    Run the full end-to-end NLP pipeline on a tax document (PDF or plain text).

    Pipeline steps:
      1. parse_document(file_path)
      2. summarize_document(text)
      3. extract_financial_entities(text)
      4. TaxQA().ask(...) for each extracted deduction section
    """
    from nlp_pipeline.src.insight import generate_insights  # noqa: PLC0415

    try:
        return generate_insights(file_path)
    except Exception as exc:
        raise ProviderError(
            "Insight pipeline failure",
            f"Failed to run document insight pipeline: {type(exc).__name__}: {str(exc)}",
        ) from exc
