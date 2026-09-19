"""
Pydantic schemas for standardized API error responses.
"""

from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel, Field


class ErrorDetail(BaseModel):
    """
    Structured error information including a machine-readable code,
    human-readable message, and optional contextual details.
    """

    code: str = Field(
        ...,
        description="Machine-readable error code (e.g. 'BAD_REQUEST', 'UNSUPPORTED_FILE_TYPE', 'FILE_TOO_LARGE', 'VALIDATION_ERROR', 'PROVIDER_ERROR', 'INTERNAL_SERVER_ERROR')",
        examples=["UNSUPPORTED_FILE_TYPE"],
    )
    message: str = Field(
        ...,
        description="Human-readable explanation of the error",
        examples=["Unsupported file type"],
    )
    details: Optional[Any] = Field(
        None,
        description="Optional additional context, parameter details, or validation failure messages",
        examples=["'.exe' is not supported. Allowed: .csv, .md, .pdf, .txt"],
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "code": "UNSUPPORTED_FILE_TYPE",
                "message": "Unsupported file type",
                "details": "'.exe' is not supported. Allowed: .csv, .md, .pdf, .txt",
            }
        }
    }


class ErrorResponse(BaseModel):
    """
    Standardized API error response returned for all 4xx and 5xx responses.
    """

    success: bool = Field(
        False,
        description="Indicates operation outcome. Always false for error responses.",
    )
    error: ErrorDetail = Field(
        ...,
        description="Structured error details",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "success": False,
                "error": {
                    "code": "UNSUPPORTED_FILE_TYPE",
                    "message": "Unsupported file type",
                    "details": "'.exe' is not supported. Allowed: .csv, .md, .pdf, .txt",
                },
            }
        }
    }
