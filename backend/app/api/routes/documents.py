"""
Document management routes with database persistence.

Endpoints:
- POST   /api/v1/documents/upload: Upload, validate, extract, and persist metadata
- GET    /api/v1/documents:        List previously processed documents
- GET    /api/v1/documents/{id}:   Retrieve specific document and associated summaries
- DELETE /api/v1/documents/{id}:   Delete document and associated summaries
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from sqlalchemy import desc, func, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, selectinload

from backend.app.api.deps import get_current_user
from backend.app.core.config import settings
from backend.app.db.models import Document, User
from backend.app.db.session import get_db
from backend.app.schemas.document import (
    DocumentDeleteResponse,
    DocumentDetailResponse,
    DocumentListResponse,
    DocumentResponse,
    DocumentUploadResponse,
    SummaryItemResponse,
)
from backend.app.schemas.error import ErrorResponse
from backend.app.services.document_service import extract_document_text, _PREVIEW_CHARS

router = APIRouter(prefix="/documents", tags=["Documents"])


def _document_not_found(document_id: str) -> HTTPException:
    """HTTP 404 for missing *and* cross-user access (does not reveal existence)."""
    return HTTPException(
        status_code=404,
        detail={
            "code": "DOCUMENT_NOT_FOUND",
            "message": f"Document with ID '{document_id}' not found.",
            "details": None,
        },
    )


def _get_owned_document(
    db: Session,
    document_id: str,
    user_id: str,
    *,
    with_summaries: bool = False,
) -> Document:
    stmt = select(Document).where(Document.id == document_id, Document.user_id == user_id)
    if with_summaries:
        stmt = stmt.options(selectinload(Document.summaries))
    doc = db.scalars(stmt).first()
    if doc is None:
        raise _document_not_found(document_id)
    return doc


@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
    responses={
        200: {"model": DocumentUploadResponse, "description": "Document parsed, text extracted, and metadata saved"},
        400: {"model": ErrorResponse, "description": "Validation error (unsupported format, empty file, invalid PDF header)"},
        401: {"model": ErrorResponse, "description": "Missing or invalid JWT"},
        413: {"model": ErrorResponse, "description": "File exceeds size limit"},
        422: {"model": ErrorResponse, "description": "Unprocessable document content (empty or insufficient text, corrupted file)"},
        500: {"model": ErrorResponse, "description": "Extraction or database persistence failed unexpectedly"},
    },
    summary="Upload a legal document",
    description=(
        "Upload a legal or tax document for validation, text extraction, and database persistence.\n\n"
        "**Supported formats**: `.pdf`, `.txt`, `.md`, `.csv`\n\n"
        f"**Size limit**: configurable via `UPLOAD_MAX_SIZE_MB` env var "
        f"(default: {settings.UPLOAD_MAX_SIZE_MB} MB)\n\n"
        "The endpoint validates the file, extracts text using the NLP parser, and "
        "persists the document metadata in PostgreSQL owned by the authenticated user. "
        "Temporary files are deleted immediately after extraction. Requires a Bearer JWT."
    ),
)
async def upload_document(
    file: UploadFile = File(..., description="Legal document file (.pdf, .txt, .md, .csv)"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> DocumentUploadResponse:
    """
    Validate and extract text from an uploaded legal document, then persist metadata in PostgreSQL.
    """
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
        print(f"[upload] unexpected extraction error: {type(exc).__name__}: {exc}")
        raise HTTPException(
            status_code=500,
            detail={
                "code": "EXTRACTION_FAILED",
                "message": "Extraction failed",
                "details": "An unexpected error occurred while processing the document.",
            },
        )

    # Persist document metadata in PostgreSQL
    try:
        db_doc = Document(
            id=doc.document_id,
            filename=doc.filename,
            file_type=doc.mime_type,
            file_size=doc.file_size_bytes,
            processing_status="completed",
            user_id=current_user.id,
        )
        db.add(db_doc)
        db.commit()
        db.refresh(db_doc)
    except SQLAlchemyError as exc:
        db.rollback()
        print(f"[upload] database error while persisting document: {type(exc).__name__}")
        raise HTTPException(
            status_code=500,
            detail={
                "code": "DATABASE_ERROR",
                "message": "Database error",
                "details": "Failed to persist document record.",
            },
        ) from exc

    preview = doc.text[:_PREVIEW_CHARS].strip()
    return DocumentUploadResponse(
        success=True,
        document_id=doc.document_id,
        filename=doc.filename,
        file_size_bytes=doc.file_size_bytes,
        mime_type=doc.mime_type,
        char_count=doc.char_count,
        preview=preview,
        message="Document uploaded and text extracted successfully",
    )


@router.get(
    "",
    response_model=DocumentListResponse,
    responses={
        200: {"model": DocumentListResponse, "description": "List of persisted document records"},
        401: {"model": ErrorResponse, "description": "Missing or invalid JWT"},
        500: {"model": ErrorResponse, "description": "Database query failure"},
    },
    summary="List uploaded documents",
    description=(
        "Retrieve history of previously uploaded legal document metadata for the "
        "authenticated user only.\n\n"
        "Returns document_id and metadata only (no full summary text). "
        "Use GET /api/v1/documents/{document_id} to load associated summaries.\n\n"
        "Cross-user documents are never included."
    ),
)
def list_documents(
    skip: int = Query(0, ge=0, description="Pagination offset"),
    limit: int = Query(100, ge=1, le=500, description="Max number of records to return"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> DocumentListResponse:
    """
    List persisted document records owned by the current user, newest first.
    """
    try:
        owner_filter = Document.user_id == current_user.id
        stmt = (
            select(Document)
            .where(owner_filter)
            .order_by(desc(Document.created_at))
            .offset(skip)
            .limit(limit)
        )
        docs = db.scalars(stmt).all()

        total = db.scalar(select(func.count()).select_from(Document).where(owner_filter)) or 0

        return DocumentListResponse(
            success=True,
            total=total,
            documents=[
                DocumentResponse(
                    document_id=doc.id,
                    filename=doc.filename,
                    file_type=doc.file_type,
                    file_size=doc.file_size,
                    processing_status=doc.processing_status,
                    created_at=doc.created_at,
                    updated_at=doc.updated_at,
                )
                for doc in docs
            ],
        )
    except SQLAlchemyError as exc:
        print(f"[list_documents] database error: {type(exc).__name__}")
        raise HTTPException(
            status_code=500,
            detail={
                "code": "DATABASE_ERROR",
                "message": "Database query failed",
                "details": "An error occurred while retrieving document history.",
            },
        ) from exc


@router.get(
    "/{document_id}",
    response_model=DocumentDetailResponse,
    responses={
        200: {"model": DocumentDetailResponse, "description": "Document details with associated summaries"},
        401: {"model": ErrorResponse, "description": "Missing or invalid JWT"},
        404: {"model": ErrorResponse, "description": "Document not found or not owned by the current user"},
        500: {"model": ErrorResponse, "description": "Database query failure"},
    },
    summary="Get document details",
    description=(
        "Retrieve a specific document record and its associated summaries by document ID. "
        "Documents owned by another user return HTTP 404 so their existence is not revealed."
    ),
)
def get_document(
    document_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> DocumentDetailResponse:
    """
    Retrieve an owned document record and any generated summaries.
    """
    try:
        doc = _get_owned_document(db, document_id, current_user.id, with_summaries=True)

        return DocumentDetailResponse(
            document_id=doc.id,
            filename=doc.filename,
            file_type=doc.file_type,
            file_size=doc.file_size,
            processing_status=doc.processing_status,
            created_at=doc.created_at,
            updated_at=doc.updated_at,
            summaries=[
                SummaryItemResponse(
                    id=item.id,
                    document_id=item.document_id,
                    summary=item.summary,
                    processing_time=item.processing_time,
                    created_at=item.created_at,
                )
                for item in (doc.summaries or [])
            ],
        )

    except HTTPException:
        raise
    except SQLAlchemyError as exc:
        print(f"[get_document] database error: {type(exc).__name__}")
        raise HTTPException(
            status_code=500,
            detail={
                "code": "DATABASE_ERROR",
                "message": "Database query failed",
                "details": "An error occurred while retrieving document details.",
            },
        ) from exc


@router.delete(
    "/{document_id}",
    response_model=DocumentDeleteResponse,
    responses={
        200: {"model": DocumentDeleteResponse, "description": "Document and associated summaries successfully deleted"},
        401: {"model": ErrorResponse, "description": "Missing or invalid JWT"},
        404: {"model": ErrorResponse, "description": "Document not found or not owned by the current user"},
        500: {"model": ErrorResponse, "description": "Database deletion failure"},
    },
    summary="Delete document",
    description=(
        "Delete an owned document and all its associated summaries. "
        "Another user's document_id returns HTTP 404."
    ),
)
def delete_document(
    document_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> DocumentDeleteResponse:
    """
    Delete an owned document and cascade-delete its summaries.
    """
    try:
        doc = _get_owned_document(db, document_id, current_user.id)

        db.delete(doc)
        db.commit()

        return DocumentDeleteResponse(
            success=True,
            document_id=document_id,
            message="Document and associated summaries deleted successfully",
        )

    except HTTPException:
        raise
    except SQLAlchemyError as exc:
        db.rollback()
        print(f"[delete_document] database error: {type(exc).__name__}")
        raise HTTPException(
            status_code=500,
            detail={
                "code": "DATABASE_ERROR",
                "message": "Database deletion failed",
                "details": "An error occurred while deleting the document.",
            },
        ) from exc
