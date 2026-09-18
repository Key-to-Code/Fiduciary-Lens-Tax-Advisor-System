"""
Document summarization route.

POST /api/v1/summarize
    Accept a legal document (multipart/form-data),
    extract text using the existing document parser,
    summarize it via the existing NLP pipeline,
    and return structured legal summary with timing metadata.
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile

from backend.app.schemas.document import ErrorDetail
from backend.app.schemas.summarize import SummarizeResponse
from backend.app.services.document_service import extract_document_text
from backend.app.services.rag_service import ProviderError, summarize_legal_document

router = APIRouter(tags=["Summarization"])


@router.post(
    "/summarize",
    response_model=SummarizeResponse,
    responses={
        400: {"model": ErrorDetail, "description": "Validation error (bad file type, empty file, invalid PDF header)"},
        413: {"model": ErrorDetail, "description": "File exceeds size limit"},
        422: {"model": ErrorDetail, "description": "Unprocessable content (empty extracted text, insufficient text, corrupted document)"},
        502: {"model": ErrorDetail, "description": "LLM or RAG provider communication failure"},
        500: {"model": ErrorDetail, "description": "Unexpected server error"},
    },
    summary="Summarize a legal document",
    description=(
        "Upload a legal or tax document (.pdf, .txt, .md, .csv) for text extraction and "
        "expert Chartered Accountant summarization using the project's NLP pipeline.\n\n"
        "The endpoint extracts document text using the existing parser, validates "
        "content adequacy, passes the text to the configured LLM, and returns "
        "the summary with processing metrics."
    ),
)
async def summarize_document_endpoint(
    file: UploadFile = File(..., description="Legal document file (.pdf, .txt, .md, .csv)"),
    provider: Optional[str] = Query(
        None,
        description="Optional LLM provider override ('openai', 'ollama', 'local', 'extractive', 'auto')",
    ),
) -> SummarizeResponse:
    """
    Validate, extract text from, and summarize an uploaded legal document.

    - Validates file type, size, and integrity.
    - Reuses existing PyMuPDF / text parser.
    - Invokes existing NLP summarization engine.
    - Does not persist the file or leak internal paths.
    """
    # ── 1. Document Extraction & Validation ───────────────────────────────────
    try:
        doc = await extract_document_text(file)
    except ValueError as exc:
        title, detail = (exc.args[0], exc.args[1]) if len(exc.args) >= 2 else (str(exc), None)
        # Content validation issues (empty text, insufficient text, corrupted file) -> 422
        # Format/input validation issues (unsupported type, empty file, bad header) -> 400
        is_content_error = any(
            k in title.lower() for k in ("empty document content", "insufficient", "corrupted")
        )
        status_code = 422 if is_content_error else 400
        raise HTTPException(
            status_code=status_code,
            detail=ErrorDetail(error=title, detail=detail).model_dump(),
        )
    except RuntimeError as exc:
        title, detail = (exc.args[0], exc.args[1]) if len(exc.args) >= 2 else (str(exc), None)
        is_too_large = "too large" in (title or "").lower()
        raise HTTPException(
            status_code=413 if is_too_large else 422,
            detail=ErrorDetail(error=title, detail=detail).model_dump(),
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=ErrorDetail(
                error="Extraction error",
                detail="An unexpected error occurred while extracting text from the document.",
            ).model_dump(),
        )

    # ── 2. NLP / RAG Summarization ───────────────────────────────────────────
    try:
        summary_result = summarize_legal_document(text=doc.text, provider=provider)
    except ProviderError as exc:
        raise HTTPException(
            status_code=502,
            detail=ErrorDetail(error=exc.title, detail=exc.detail).model_dump(),
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=ErrorDetail(
                error="Summarization failed",
                detail="An unexpected error occurred while generating the legal document summary.",
            ).model_dump(),
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
    )
