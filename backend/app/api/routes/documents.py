"""
Document upload route.

POST /api/v1/documents/upload
    Accept a legal document file (multipart/form-data).
    Validate, extract text, and return metadata + preview.
"""

from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from backend.app.schemas.document import DocumentUploadResponse, ErrorDetail
from backend.app.services.document_service import process_upload
from backend.app.core.config import settings

router = APIRouter(prefix="/documents", tags=["Documents"])


@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
    responses={
        400: {"model": ErrorDetail, "description": "Validation error (bad type, empty, invalid PDF)"},
        413: {"model": ErrorDetail, "description": "File exceeds size limit"},
        422: {"description": "Malformed request — missing or invalid field"},
        500: {"model": ErrorDetail, "description": "Extraction failed unexpectedly"},
    },
    summary="Upload a legal document",
    description=(
        "Upload a legal document for text extraction.\n\n"
        "**Supported formats**: `.pdf`, `.txt`, `.md`, `.csv`\n\n"
        f"**Size limit**: configurable via `UPLOAD_MAX_SIZE_MB` env var "
        f"(default: 20 MB)\n\n"
        "The endpoint validates the file, extracts its text using the existing "
        "NLP pipeline parser, and returns the extracted text length and a preview. "
        "The file is not persisted after extraction."
    ),
)
async def upload_document(
    file: UploadFile = File(..., description="Legal document to upload (.pdf, .txt, .md, .csv)"),
) -> DocumentUploadResponse:
    """
    Validate and extract text from an uploaded legal document.

    - Accepts **multipart/form-data** with a `file` field.
    - Returns document metadata and a preview of the extracted text.
    - The temporary file is deleted immediately after extraction.
    """
    try:
        result = await process_upload(file)
        return DocumentUploadResponse(**result)

    except ValueError as exc:
        # exc.args: (error_title, detail_message) from document_service
        title, detail = (exc.args[0], exc.args[1]) if len(exc.args) >= 2 else (str(exc), None)
        raise HTTPException(
            status_code=400,
            detail=ErrorDetail(error=title, detail=detail).model_dump(),
        )

    except RuntimeError as exc:
        # RuntimeError is used by document_service to signal 413 (too large)
        # and by parse_document to signal PDF parse failures.
        title, detail = (exc.args[0], exc.args[1]) if len(exc.args) >= 2 else (str(exc), None)
        is_too_large = "too large" in (title or "").lower()
        raise HTTPException(
            status_code=413 if is_too_large else 422,
            detail=ErrorDetail(error=title, detail=detail).model_dump(),
        )

    except Exception as exc:  # noqa: BLE001
        # Catch-all — log but do not expose internal details to the client.
        print(f"[upload] unexpected error: {type(exc).__name__}: {exc}")
        raise HTTPException(
            status_code=500,
            detail=ErrorDetail(
                error="Extraction failed",
                detail="An unexpected error occurred while processing the document.",
            ).model_dump(),
        )
