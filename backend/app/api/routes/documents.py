"""
Document upload route.

POST /api/v1/documents/upload
    Accept a legal document file (multipart/form-data).
    Validate, extract text, and return metadata + preview.
"""

from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, UploadFile

from backend.app.schemas.document import DocumentUploadResponse
from backend.app.schemas.error import ErrorResponse
from backend.app.services.document_service import process_upload
from backend.app.core.config import settings

router = APIRouter(prefix="/documents", tags=["Documents"])


@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
    responses={
        200: {"model": DocumentUploadResponse, "description": "Document parsed and text extracted successfully"},
        400: {"model": ErrorResponse, "description": "Validation error (unsupported format, empty file, invalid PDF header)"},
        413: {"model": ErrorResponse, "description": "File exceeds size limit"},
        422: {"model": ErrorResponse, "description": "Unprocessable document content (empty or insufficient text, corrupted file)"},
        500: {"model": ErrorResponse, "description": "Extraction failed unexpectedly"},
    },
    summary="Upload a legal document",
    description=(
        "Upload a legal or tax document for validation and text extraction.\n\n"
        "**Supported formats**: `.pdf`, `.txt`, `.md`, `.csv`\n\n"
        f"**Size limit**: configurable via `UPLOAD_MAX_SIZE_MB` env var "
        f"(default: {settings.UPLOAD_MAX_SIZE_MB} MB)\n\n"
        "The endpoint validates the file, extracts its text using the project's "
        "NLP pipeline parser, and returns the extracted text length and a preview. "
        "The file is not persisted on the filesystem after extraction."
    ),
)
async def upload_document(
    file: UploadFile = File(..., description="Legal document file (.pdf, .txt, .md, .csv)"),
) -> DocumentUploadResponse:
    """
    Validate and extract text from an uploaded legal document.

    - Accepts **multipart/form-data** with a `file` field.
    - Returns document metadata and a preview of the extracted text.
    - Temporary files are deleted immediately after extraction in a `finally` block.
    """
    try:
        result = await process_upload(file)
        return DocumentUploadResponse(**result)

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
        title_lower = (title or "").lower()

        if "too large" in title_lower:
            raise HTTPException(
                status_code=413,
                detail={"code": "FILE_TOO_LARGE", "message": title, "details": detail},
            )
        if "unavailable" in title_lower:
            raise HTTPException(
                status_code=503,
                detail={"code": "DEPENDENCY_ERROR", "message": title, "details": detail},
            )

        raise HTTPException(
            status_code=422,
            detail={"code": "PROCESSING_ERROR", "message": title, "details": detail},
        )

    except Exception as exc:  # noqa: BLE001
        print(f"[upload] unexpected error: {type(exc).__name__}: {exc}")
        raise HTTPException(
            status_code=500,
            detail={
                "code": "EXTRACTION_FAILED",
                "message": "Extraction failed",
                "details": "An unexpected error occurred while processing the document.",
            },
        )
