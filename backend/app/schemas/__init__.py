"""
Central schema exports for the Fiduciary Lens API.
"""

from backend.app.schemas.auth import LoginRequest, RegisterRequest, TokenResponse, UserResponse
from backend.app.schemas.document import (
    DocumentDeleteResponse,
    DocumentDetailResponse,
    DocumentListResponse,
    DocumentResponse,
    DocumentUploadResponse,
    SummaryItemResponse,
)
from backend.app.schemas.error import ErrorDetail, ErrorResponse
from backend.app.schemas.health import HealthResponse
from backend.app.schemas.summarize import SummarizeRequest, SummarizeResponse

__all__ = [
    "RegisterRequest",
    "LoginRequest",
    "TokenResponse",
    "UserResponse",
    "DocumentUploadResponse",
    "DocumentResponse",
    "DocumentDetailResponse",
    "DocumentListResponse",
    "DocumentDeleteResponse",
    "SummaryItemResponse",
    "ErrorDetail",
    "ErrorResponse",
    "HealthResponse",
    "SummarizeRequest",
    "SummarizeResponse",
]
