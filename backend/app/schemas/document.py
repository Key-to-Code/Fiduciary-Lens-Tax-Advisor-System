"""
Pydantic schemas for the document upload endpoint.
"""

from __future__ import annotations

from pydantic import BaseModel, Field
from backend.app.schemas.error import ErrorDetail, ErrorResponse


class DocumentUploadResponse(BaseModel):
    """Successful response body for POST /api/v1/documents/upload."""

    success: bool = Field(
        True,
        description="Indicates whether document upload and parsing was successful",
    )
    document_id: str = Field(
        ...,
        description="Unique processing identifier generated for this document",
        examples=["550e8400-e29b-41d4-a716-446655440000"],
    )
    filename: str = Field(
        ...,
        description="Sanitized display filename (never an internal filesystem path)",
        examples=["form16.pdf"],
    )
    file_size_bytes: int = Field(
        ...,
        ge=0,
        description="Size of the uploaded document in bytes",
        examples=[634075],
    )
    mime_type: str = Field(
        ...,
        description="MIME type detected for the uploaded document",
        examples=["application/pdf"],
    )
    char_count: int = Field(
        ...,
        ge=0,
        description="Number of text characters extracted from the document",
        examples=[12340],
    )
    preview: str = Field(
        ...,
        description="Initial text preview extracted from the document (up to 500 characters)",
        examples=["FORM 16 — Certificate under section 203 of the Income-tax Act..."],
    )
    message: str = Field(
        "Document uploaded and text extracted successfully",
        description="Human-readable status message",
    )

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


__all__ = ["DocumentUploadResponse", "ErrorDetail", "ErrorResponse"]
