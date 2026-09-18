"""
Pydantic response schemas for the health endpoint.
"""

from pydantic import BaseModel


class HealthResponse(BaseModel):
    """Response body returned by GET /health."""

    status: str
    service: str

    model_config = {"json_schema_extra": {"example": {"status": "healthy", "service": "legal-document-summarizer"}}}
