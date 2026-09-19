"""
Document summarization route.

POST /api/v1/summarize
    Accept a legal document (multipart/form-data) or JSON body,
    extract text using the existing document parser,
    summarize it via the existing NLP pipeline,
    and return structured legal summary with timing metadata.

POST /api/v1/summarize/text
    Accept raw text payload (application/json) using SummarizeRequest,
    summarize it via the existing NLP pipeline,
    and return structured legal summary with timing metadata.
"""

from __future__ import annotations

import uuid
from typing import Optional

from fastapi import APIRouter, File, HTTPException, Query, Request, UploadFile

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


@router.post(
    "/summarize",
    response_model=SummarizeResponse,
    responses={
        200: {"model": SummarizeResponse, "description": "Document summarized successfully"},
        400: {"model": ErrorResponse, "description": "Validation error (unsupported file type, empty file, invalid provider)"},
        413: {"model": ErrorResponse, "description": "File exceeds upload size limit"},
        422: {"model": ErrorResponse, "description": "Unprocessable content (empty extracted text, insufficient text, corrupted document)"},
        500: {"model": ErrorResponse, "description": "Unexpected server error"},
        502: {"model": ErrorResponse, "description": "LLM or RAG provider communication failure"},
    },
    summary="Summarize a legal document",
    description=(
        "Upload a legal or tax document (.pdf, .txt, .md, .csv) for text extraction and "
        "expert Chartered Accountant summarization using the project's NLP pipeline.\n\n"
        "The endpoint extracts document text using the existing parser, validates "
        "content adequacy, passes the text to the configured LLM, and returns "
        "the summary with processing metrics.\n\n"
        "Also supports raw text JSON payloads sent directly to this endpoint."
    ),
)
async def summarize_document_endpoint(
    request: Request,
    file: Optional[UploadFile] = File(
        None,
        description="Legal document file (.pdf, .txt, .md, .csv). Optional if submitting raw JSON text.",
    ),
    provider: Optional[str] = Query(
        None,
        description="Optional LLM provider override ('openai', 'ollama', 'local', 'extractive', 'auto')",
    ),
) -> SummarizeResponse:
    """
    Validate, extract text from, and summarize an uploaded legal document or direct text payload.

    - Validates file type, size, and content integrity.
    - Reuses existing PyMuPDF / text parser.
    - Invokes existing NLP summarization engine.
    - Does not persist the file or leak internal paths.
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

        try:
            summary_result = summarize_legal_document(text=text, provider=target_provider)
        except ProviderError as exc:
            raise HTTPException(
                status_code=502,
                detail={"code": "PROVIDER_ERROR", "message": exc.title, "details": exc.detail},
            ) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail={
                    "code": "SUMMARIZATION_FAILED",
                    "message": "Summarization failed",
                    "details": "An unexpected error occurred while generating the legal document summary.",
                },
            ) from exc

        encoded_len = len(text.encode("utf-8"))
        return SummarizeResponse(
            success=True,
            filename=doc_name,
            document_id=str(uuid.uuid4()),
            file_size_bytes=encoded_len,
            char_count=len(text),
            summary=summary_result["summary"],
            processing_time=summary_result["processing_time"],
            message="Document summarized successfully",
            provider=target_provider,
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

    # 2. NLP / RAG Summarization
    try:
        summary_result = summarize_legal_document(text=doc.text, provider=cleaned_provider)
    except ProviderError as exc:
        raise HTTPException(
            status_code=502,
            detail={"code": "PROVIDER_ERROR", "message": exc.title, "details": exc.detail},
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail={
                "code": "SUMMARIZATION_FAILED",
                "message": "Summarization failed",
                "details": "An unexpected error occurred while generating the legal document summary.",
            },
        )

    return SummarizeResponse(
        success=True,
        filename=doc.filename,
        document_id=doc.document_id,
        file_size_bytes=doc.file_size_bytes,
        char_count=doc.char_count,
        summary=summary_result["summary"],
        processing_time=summary_result["processing_time"],
        message="Document summarized successfully",
        provider=cleaned_provider,
    )


@router.post(
    "/summarize/text",
    response_model=SummarizeResponse,
    responses={
        200: {"model": SummarizeResponse, "description": "Text summarized successfully"},
        400: {"model": ErrorResponse, "description": "Invalid input or provider"},
        422: {"model": ErrorResponse, "description": "Request validation error (e.g., text too short)"},
        500: {"model": ErrorResponse, "description": "Unexpected server error"},
        502: {"model": ErrorResponse, "description": "LLM or RAG provider failure"},
    },
    summary="Summarize raw legal text",
    description=(
        "Submit raw text of a legal document or tax notice in JSON format for "
        "Chartered Accountant summarization using the configured NLP pipeline."
    ),
)
async def summarize_text_endpoint(
    payload: SummarizeRequest,
) -> SummarizeResponse:
    """
    Summarize raw legal text passed in a JSON body via SummarizeRequest.
    """
    target_provider = _validate_provider_str(payload.provider)

    try:
        summary_result = summarize_legal_document(text=payload.text, provider=target_provider)
    except ProviderError as exc:
        raise HTTPException(
            status_code=502,
            detail={"code": "PROVIDER_ERROR", "message": exc.title, "details": exc.detail},
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail={
                "code": "SUMMARIZATION_FAILED",
                "message": "Summarization failed",
                "details": "An unexpected error occurred while generating the legal document summary.",
            },
        ) from exc

    doc_name = payload.document_name or "text_input.txt"
    encoded_len = len(payload.text.encode("utf-8"))

    return SummarizeResponse(
        success=True,
        filename=doc_name,
        document_id=str(uuid.uuid4()),
        file_size_bytes=encoded_len,
        char_count=len(payload.text),
        summary=summary_result["summary"],
        processing_time=summary_result["processing_time"],
        message="Document summarized successfully",
        provider=target_provider,
    )
