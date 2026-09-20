"""
Document summarization route with database persistence.

POST /api/v1/summarize
    Accept a legal document (multipart/form-data) or JSON body,
    extract text using the existing document parser,
    summarize via the existing NLP pipeline, and persist the summary.

    Optional form/JSON field ``document_id``:
      CASE A — supplied: attach summary to that existing Document row
               (do not create another document).
      CASE B — omitted: create a new Document row (backward compatible).

POST /api/v1/summarize/text
    Accept raw text payload (application/json) using SummarizeRequest,
    with the same optional document_id CASE A/B behavior.
"""

from __future__ import annotations

import uuid
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, UploadFile
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from backend.app.db.models import Document, Summary
from backend.app.db.session import get_db
from backend.app.schemas.error import ErrorResponse
from backend.app.schemas.summarize import SummarizeRequest, SummarizeResponse
from backend.app.services.document_service import extract_document_text
from backend.app.services.rag_service import ProviderError, summarize_legal_document

router = APIRouter(tags=["Summarization"])

_ALLOWED_PROVIDERS = frozenset({"openai", "ollama", "local", "extractive", "auto"})


def _validate_provider_str(provider: Optional[str]) -> Optional[str]:
    """Validate optional provider string, raising 400 for unknown values."""
    if provider is None:
        return None
    cleaned = provider.strip().lower()
    if cleaned not in _ALLOWED_PROVIDERS:
        allowed = ", ".join(f"'{p}'" for p in sorted(_ALLOWED_PROVIDERS))
        raise HTTPException(
            status_code=400,
            detail={
                "code": "INVALID_PROVIDER",
                "message": f"Unsupported provider '{provider}'.",
                "details": f"Allowed providers: {allowed}",
            },
        )
    return cleaned


def _normalize_optional_document_id(value: Optional[str]) -> Optional[str]:
    """Treat missing/blank document_id as omitted (backward-compatible create path)."""
    if value is None:
        return None
    cleaned = value.strip()
    return cleaned or None


def _persist_and_summarize(
    db: Session,
    text: str,
    filename: str,
    file_size_bytes: int,
    char_count: int,
    mime_type: str,
    provider: Optional[str],
    doc_id: Optional[str] = None,
    require_existing: bool = False,
) -> SummarizeResponse:
    """
    Persist document metadata (or reuse an existing row), invoke the NLP
    summarization engine, and store the resulting summary in PostgreSQL.

    If ``require_existing`` is True, ``doc_id`` must refer to an existing
    Document row. No additional Document row is created.
    If ``require_existing`` is False, a new Document row is created when
    ``doc_id`` is absent or not already present.
    """
    target_id = _normalize_optional_document_id(doc_id)

    if require_existing:
        if not target_id:
            raise HTTPException(
                status_code=400,
                detail={
                    "code": "INVALID_DOCUMENT_ID",
                    "message": "document_id is required when attaching a summary to an existing document.",
                    "details": None,
                },
            )
        existing_doc = db.get(Document, target_id)
        if existing_doc is None:
            raise HTTPException(
                status_code=404,
                detail={
                    "code": "DOCUMENT_NOT_FOUND",
                    "message": f"Document with ID '{target_id}' not found.",
                    "details": "Summarization did not create a new document because document_id was supplied.",
                },
            )
        db_doc = existing_doc
        try:
            db_doc.processing_status = "processing"
            db.commit()
            db.refresh(db_doc)
        except SQLAlchemyError as exc:
            db.rollback()
            print(f"[summarize] database error while updating existing document: {type(exc).__name__}")
            raise HTTPException(
                status_code=500,
                detail={
                    "code": "DATABASE_ERROR",
                    "message": "Database error",
                    "details": "Failed to update existing document record for summarization.",
                },
            ) from exc
    else:
        target_id = target_id or str(uuid.uuid4())
        try:
            existing_doc = db.get(Document, target_id)
            if existing_doc:
                db_doc = existing_doc
                db_doc.processing_status = "processing"
            else:
                db_doc = Document(
                    id=target_id,
                    filename=filename,
                    file_type=mime_type,
                    file_size=file_size_bytes,
                    processing_status="processing",
                )
                db.add(db_doc)
            db.commit()
            db.refresh(db_doc)
        except SQLAlchemyError as exc:
            db.rollback()
            print(f"[summarize] database error while creating document: {type(exc).__name__}")
            raise HTTPException(
                status_code=500,
                detail={
                    "code": "DATABASE_ERROR",
                    "message": "Database error",
                    "details": "Failed to initialize document record for summarization.",
                },
            ) from exc

    # 2. Invoke NLP summarization pipeline
    try:
        summary_result = summarize_legal_document(text=text, provider=provider)
    except ProviderError as exc:
        try:
            db_doc.processing_status = "failed"
            db.commit()
        except SQLAlchemyError:
            db.rollback()
        raise HTTPException(
            status_code=502,
            detail={"code": "PROVIDER_ERROR", "message": exc.title, "details": exc.detail},
        ) from exc
    except Exception as exc:
        try:
            db_doc.processing_status = "failed"
            db.commit()
        except SQLAlchemyError:
            db.rollback()
        raise HTTPException(
            status_code=500,
            detail={
                "code": "SUMMARIZATION_FAILED",
                "message": "Summarization failed",
                "details": "An unexpected error occurred while generating the legal document summary.",
            },
        ) from exc

    # 3. Persist generated Summary record and update document status
    try:
        db_summary = Summary(
            id=str(uuid.uuid4()),
            document_id=target_id,
            summary=summary_result["summary"],
            processing_time=summary_result["processing_time"],
        )
        db_doc.processing_status = "completed"
        db.add(db_summary)
        db.commit()
        db.refresh(db_doc)
    except SQLAlchemyError as exc:
        db.rollback()
        print(f"[summarize] database error while saving summary: {type(exc).__name__}")
        raise HTTPException(
            status_code=500,
            detail={
                "code": "DATABASE_ERROR",
                "message": "Database error",
                "details": "Failed to persist document summary.",
            },
        ) from exc

    return SummarizeResponse(
        success=True,
        filename=filename,
        document_id=target_id,
        file_size_bytes=file_size_bytes,
        char_count=char_count,
        summary=summary_result["summary"],
        processing_time=summary_result["processing_time"],
        message="Document summarized successfully",
        provider=provider,
    )


@router.post(
    "/summarize",
    response_model=SummarizeResponse,
    responses={
        200: {"model": SummarizeResponse, "description": "Document summarized and persisted successfully"},
        400: {"model": ErrorResponse, "description": "Validation error (unsupported file type, empty file, invalid provider)"},
        413: {"model": ErrorResponse, "description": "File exceeds upload size limit"},
        422: {"model": ErrorResponse, "description": "Unprocessable content (empty extracted text, insufficient text, corrupted document)"},
        500: {"model": ErrorResponse, "description": "Unexpected server or database error"},
        502: {"model": ErrorResponse, "description": "LLM or RAG provider communication failure"},
    },
    summary="Summarize a legal document",
    description=(
        "Upload a legal or tax document (.pdf, .txt, .md, .csv) for text extraction, "
        "expert Chartered Accountant summarization, and database persistence.\n\n"
        "The endpoint extracts document text using the existing parser, validates "
        "content adequacy, passes the text to the configured LLM, and persists "
        "the generated summary in PostgreSQL.\n\n"
        "**Document identity**\n\n"
        "- **CASE A** — optional form field `document_id` is supplied: the existing "
        "Document row is reused. No new document is created. Identity is the UUID, "
        "not the filename.\n"
        "- **CASE B** — `document_id` is omitted: a new Document row is created "
        "(backward-compatible default).\n\n"
        "Also supports raw text JSON payloads sent directly to this endpoint. "
        "JSON may include optional `document_id` with the same CASE A/B behavior."
    ),
)
async def summarize_document_endpoint(
    request: Request,
    file: Optional[UploadFile] = File(
        None,
        description="Legal document file (.pdf, .txt, .md, .csv). Optional if submitting raw JSON text.",
    ),
    document_id: Optional[str] = Form(
        None,
        description=(
            "Optional existing document UUID. When provided, summarize against that "
            "document and do not create a new Document row. When omitted, a new "
            "document record is created."
        ),
    ),
    provider: Optional[str] = Query(
        None,
        description="Optional LLM provider override ('openai', 'ollama', 'local', 'extractive', 'auto')",
    ),
    db: Session = Depends(get_db),
) -> SummarizeResponse:
    """
    Validate, extract text from, summarize, and persist an uploaded legal document or direct JSON payload.
    """
    cleaned_provider = _validate_provider_str(provider)
    content_type = request.headers.get("content-type", "")

    # ── Branch A: Direct JSON payload sent to /summarize ───────────────────────
    if "application/json" in content_type:
        try:
            body = await request.json()
            req_model = SummarizeRequest.model_validate(body)
        except Exception as exc:
            raise HTTPException(
                status_code=422,
                detail={
                    "code": "VALIDATION_ERROR",
                    "message": "Invalid JSON payload for summarization request.",
                    "details": str(exc),
                },
            ) from exc

        target_provider = _validate_provider_str(req_model.provider) or cleaned_provider
        doc_name = req_model.document_name or "document.txt"
        text = req_model.text
        encoded_len = len(text.encode("utf-8"))
        existing_id = _normalize_optional_document_id(req_model.document_id)

        return _persist_and_summarize(
            db=db,
            text=text,
            filename=doc_name,
            file_size_bytes=encoded_len,
            char_count=len(text),
            mime_type="text/plain",
            provider=target_provider,
            doc_id=existing_id,
            require_existing=existing_id is not None,
        )

    # ── Branch B: File Upload (multipart/form-data) ───────────────────────────
    if file is None:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "VALIDATION_ERROR",
                "message": "A document file (multipart/form-data) or JSON body with 'text' is required.",
                "details": "Missing 'file' field in multipart form-data.",
            },
        )

    # 1. Document Extraction & Validation
    try:
        doc = await extract_document_text(file)
    except ValueError as exc:
        title, detail = (exc.args[0], exc.args[1]) if len(exc.args) >= 2 else (str(exc), None)
        title_lower = (title or "").lower()

        if "unsupported" in title_lower:
            code = "UNSUPPORTED_FILE_TYPE"
            status_code = 400
        elif "empty file" in title_lower:
            code = "EMPTY_FILE"
            status_code = 400
        elif "invalid pdf" in title_lower:
            code = "INVALID_PDF"
            status_code = 400
        elif any(k in title_lower for k in ("empty document content", "insufficient")):
            code = "INSUFFICIENT_CONTENT"
            status_code = 422
        elif "corrupted" in title_lower:
            code = "CORRUPTED_DOCUMENT"
            status_code = 422
        else:
            code = "BAD_REQUEST"
            status_code = 400

        raise HTTPException(
            status_code=status_code,
            detail={"code": code, "message": title, "details": detail},
        )
    except RuntimeError as exc:
        title, detail = (exc.args[0], exc.args[1]) if len(exc.args) >= 2 else (str(exc), None)
        is_too_large = "too large" in (title or "").lower()
        raise HTTPException(
            status_code=413 if is_too_large else 422,
            detail={
                "code": "FILE_TOO_LARGE" if is_too_large else "PROCESSING_ERROR",
                "message": title,
                "details": detail,
            },
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail={
                "code": "EXTRACTION_FAILED",
                "message": "Extraction error",
                "details": "An unexpected error occurred while extracting text from the document.",
            },
        )

    # 2. Persist metadata and summarize (reuse uploaded document when document_id is given)
    existing_id = _normalize_optional_document_id(document_id)
    return _persist_and_summarize(
        db=db,
        text=doc.text,
        filename=doc.filename,
        file_size_bytes=doc.file_size_bytes,
        char_count=doc.char_count,
        mime_type=doc.mime_type,
        provider=cleaned_provider,
        doc_id=existing_id or doc.document_id,
        require_existing=existing_id is not None,
    )


@router.post(
    "/summarize/text",
    response_model=SummarizeResponse,
    responses={
        200: {"model": SummarizeResponse, "description": "Text summarized and persisted successfully"},
        400: {"model": ErrorResponse, "description": "Invalid input or provider"},
        422: {"model": ErrorResponse, "description": "Request validation error (e.g., text too short)"},
        500: {"model": ErrorResponse, "description": "Unexpected server or database error"},
        502: {"model": ErrorResponse, "description": "LLM or RAG provider failure"},
    },
    summary="Summarize raw legal text",
    description=(
        "Submit raw text of a legal document or tax notice in JSON format for "
        "Chartered Accountant summarization and database persistence."
    ),
)
async def summarize_text_endpoint(
    payload: SummarizeRequest,
    db: Session = Depends(get_db),
) -> SummarizeResponse:
    """
    Summarize raw legal text passed in a JSON body via SummarizeRequest and persist in PostgreSQL.
    """
    target_provider = _validate_provider_str(payload.provider)
    doc_name = payload.document_name or "text_input.txt"
    encoded_len = len(payload.text.encode("utf-8"))
    existing_id = _normalize_optional_document_id(payload.document_id)

    return _persist_and_summarize(
        db=db,
        text=payload.text,
        filename=doc_name,
        file_size_bytes=encoded_len,
        char_count=len(payload.text),
        mime_type="text/plain",
        provider=target_provider,
        doc_id=existing_id,
        require_existing=existing_id is not None,
    )
