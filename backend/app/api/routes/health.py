"""
Health-check route.

GET /api/v1/health  →  200 {"status": "healthy", "service": "legal-document-summarizer", ...}
"""

from fastapi import APIRouter
from backend.app.core.config import settings
from backend.app.schemas.health import HealthResponse

router = APIRouter(tags=["Health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Returns the current health status and metadata of the API service.",
)
async def health_check() -> HealthResponse:
    """Lightweight liveness probe — no I/O, always fast."""
    return HealthResponse(
        status="healthy",
        service="legal-document-summarizer",
        version=settings.APP_VERSION,
        environment=settings.APP_ENV,
    )
