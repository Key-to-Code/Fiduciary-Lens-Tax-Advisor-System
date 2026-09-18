"""
Pydantic schemas for the document upload endpoint.
"""

from __future__ import annotations

from pydantic import BaseModel


class DocumentUploadResponse(BaseModel):
    """Successful response body for POST /documents/upload."""

    success: bool
    document_id: str
    filename: str       # sanitised display name; never a filesystem path
    file_size_bytes: int
    mime_type: str
    char_count: int     # length of the extracted text
    preview: str        # first 500 chars of extracted text
    message: str

    model_config = {
        "json_schema_extra": {
            "example": {
                "success": True,
                "document_id": "550e8400-e29b-41d4-a716-446655440000",
                "filename": "form16.pdf",
                "file_size_bytes": 634075,
                "mime_type": "application/pdf",
                "char_count": 12340,
                "preview": "FORM 16 — Certificate under section 203 of the Income-tax Act...",
                "message": "Document uploaded and text extracted successfully",
            }
        }
    }


class ErrorDetail(BaseModel):
    """Error response body — returned for 4xx and 5xx responses."""

    success: bool = False
    error: str
    detail: str | None = None

    model_config = {
        "json_schema_extra": {
            "example": {
                "success": False,
                "error": "Unsupported file type",
                "detail": "'.docx' is not supported. Allowed: .pdf, .txt, .md, .csv",
            }
        }
    }
