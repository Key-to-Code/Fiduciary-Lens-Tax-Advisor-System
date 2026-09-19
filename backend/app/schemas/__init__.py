"""
Central schema exports for the Fiduciary Lens API.
"""

from backend.app.schemas.document import DocumentUploadResponse
from backend.app.schemas.error import ErrorDetail, ErrorResponse
from backend.app.schemas.health import HealthResponse
from backend.app.schemas.summarize import SummarizeRequest, SummarizeResponse

__all__ = [
    "DocumentUploadResponse",
    "ErrorDetail",
    "ErrorResponse",
    "HealthResponse",
    "SummarizeRequest",
    "SummarizeResponse",
]
