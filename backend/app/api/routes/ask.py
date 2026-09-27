"""
Public tax Q&A routes for the chat frontend.

POST /api/v1/ask        — wrap TaxQA without duplicating retrieval
GET  /api/v1/knowledge  — index freshness from the FAISS manifest (no model load)
"""

from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException

from shared import config
from backend.app.schemas.ask import (
    AskRequest,
    AskResponse,
    KnowledgeStatusResponse,
    SourceItem,
)
from backend.app.schemas.error import ErrorResponse
from backend.app.services.rag_service import ProviderError, ask_tax_question

router = APIRouter(tags=["Tax Q&A"])
_DEFAULT_INDEX_DIR = config.INDEX_DIR
INDEX_DIR = config.INDEX_DIR


def _knowledge_label(available: bool, n_chunks: int, documents: list[str]) -> str:
    if not available:
        return "KB offline · index not built"
    names = " · ".join(
        name.replace(".pdf", "").replace("-", " ") for name in documents[:3]
    )
    if names:
        return f"KB current · {n_chunks:,} passages · {names}"
    return f"KB current · {n_chunks:,} passages"


@router.get(
    "/knowledge",
    response_model=KnowledgeStatusResponse,
    summary="Knowledge-base status",
    description="Reads index/manifest.json only. Does not load embeddings or the LLM.",
)
def knowledge_status() -> KnowledgeStatusResponse:
    index_dir = config.INDEX_DIR if config.INDEX_DIR != _DEFAULT_INDEX_DIR else INDEX_DIR
    manifest_path = index_dir / "manifest.json"
    if not manifest_path.exists():
        return KnowledgeStatusResponse(
            available=False,
            ready=False,
            n_chunks=0,
            documents=[],
            embed_model=None,
            label=_knowledge_label(False, 0, []),
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return KnowledgeStatusResponse(
            available=False,
            ready=False,
            n_chunks=0,
            documents=[],
            embed_model=None,
            label=_knowledge_label(False, 0, []),
        )
    documents = list(manifest.get("documents") or [])
    n_chunks = int(manifest.get("n_chunks") or 0)
    return KnowledgeStatusResponse(
        available=True,
        ready=True,
        n_chunks=n_chunks,
        documents=documents,
        embed_model=manifest.get("embed_model"),
        label=_knowledge_label(True, n_chunks, documents),
    )


@router.post(
    "/ask",
    response_model=AskResponse,
    responses={
        200: {"model": AskResponse},
        422: {"model": ErrorResponse},
        502: {"model": ErrorResponse, "description": "RAG or LLM provider failure"},
        503: {"model": ErrorResponse, "description": "Index not built"},
    },
    summary="Ask a tax-law question",
    description=(
        "Public educational Q&A over the Income-tax Act, 2025 knowledge base. "
        "Wraps the existing TaxQA engine. Ungrounded retrieval returns the engine's "
        "refusal text with an empty sources list — the API never invents citations."
    ),
)
def ask_endpoint(payload: AskRequest) -> AskResponse:
    history = [(turn.question, turn.answer) for turn in payload.history]
    try:
        result = ask_tax_question(
            question=payload.question,
            provider=payload.provider,
            history=history,
        )
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail={
                "code": "INDEX_NOT_BUILT",
                "message": "Knowledge index is not built.",
                "details": str(exc),
            },
        ) from exc
    except ProviderError as exc:
        title = (exc.title or "").lower()
        detail = (exc.detail or "").lower()
        missing_index = "no index" in title or "no index" in detail or "file not found" in detail
        if missing_index:
            raise HTTPException(
                status_code=503,
                detail={
                    "code": "INDEX_NOT_BUILT",
                    "message": exc.title,
                    "details": exc.detail,
                },
            ) from exc
        raise HTTPException(
            status_code=502,
            detail={
                "code": "PROVIDER_ERROR",
                "message": exc.title,
                "details": exc.detail,
            },
        ) from exc

    sources = [SourceItem.model_validate(item) for item in result["sources"]]
    grounded = bool(result["grounded"])
    message = (
        "Answer generated from the tax knowledge base."
        if grounded
        else "No sufficiently relevant provision was retrieved."
    )
    return AskResponse(
        success=True,
        answer=result["answer"],
        sources=sources,
        grounded=grounded,
        provider=result["provider"],
        latency_ms=int(result["latency_ms"]),
        message=message,
    )
