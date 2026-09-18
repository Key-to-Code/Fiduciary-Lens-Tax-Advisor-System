"""
RAG/NLP service integration stub.

Stage 1: This module documents the call contracts for the existing pipeline
without importing the heavy ML dependencies at server start-up. Imports are
deferred inside each function so that FastAPI starts instantly even when the
FAISS index has not been built yet.

Stage 2 will:
  - Wire these functions into API route handlers.
  - Add request/response serialisation.
  - Optionally add a lazily-initialised TaxQA singleton for performance.

THE EXISTING PIPELINE IS NEVER MODIFIED HERE.
All logic stays in:
  - rag_model/src/generation/answer.py  →  TaxQA
  - nlp_pipeline/src/insight.py         →  generate_insights
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# RAG Q&A  (wraps TaxQA from rag_model)
# ---------------------------------------------------------------------------

def ask_tax_question(question: str, provider: str | None = None) -> dict[str, Any]:
    """
    Ask a single tax question against the Income-tax Act 2025 knowledge base.

    Parameters
    ----------
    question:
        The natural-language question to answer.
    provider:
        Optional LLM provider override ("openai", "ollama", "local",
        "extractive", "auto"). Defaults to the value of LLM_PROVIDER in .env.

    Returns
    -------
    dict with keys:
        answer   (str)        Full answer text including disclaimer.
        sources  (list[dict]) Cited provisions; empty when retrieval fails.
        grounded (bool)       False when retrieval confidence is too low.
        provider (str)        Which LLM backend was used.
        latency_ms (int)      Wall-clock ms for the full call.

    Raises
    ------
    FileNotFoundError
        If the FAISS index has not been built yet (run build_index.py first).
    """
    # Deferred import — keeps server startup fast; heavy models load on first call.
    from rag_model.src.generation.answer import TaxQA  # noqa: PLC0415

    qa = TaxQA(provider=provider)
    answer = qa.ask(question)
    return {
        "answer": answer.text,
        "sources": answer.sources,
        "grounded": answer.grounded,
        "provider": answer.provider,
        "latency_ms": answer.latency_ms,
    }


# ---------------------------------------------------------------------------
# Document insight pipeline  (wraps generate_insights from nlp_pipeline)
# ---------------------------------------------------------------------------

def analyse_document(file_path: str | Path) -> dict[str, Any]:
    """
    Run the end-to-end NLP pipeline on a tax document (PDF or plain text).

    Pipeline steps (all defined in the existing nlp_pipeline/):
      1. parse   — extract raw text via PyMuPDF
      2. summarize — LLM-generated summary with action items / deadlines
      3. extract — regex + LLM entity extraction (PAN, TAN, deductions …)
      4. insights — each detected deduction is queried against the RAG engine

    Parameters
    ----------
    file_path:
        Absolute or relative path to the document (.pdf / .txt / .md / .csv).

    Returns
    -------
    dict with keys:
        summary     (str)   LLM-generated document summary.
        entities    (dict)  Extracted PAN, TAN, GSTIN, amounts, complex entities.
        tax_insights (dict) Per-deduction RAG answers; key "general" if none found.

    Raises
    ------
    FileNotFoundError
        If file_path does not exist.
    ValueError
        If the file format is not supported.
    """
    # Deferred import — avoids pulling in pymupdf / transformers at startup.
    from nlp_pipeline.src.insight import generate_insights  # noqa: PLC0415

    return generate_insights(file_path)
