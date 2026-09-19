"""
Pydantic response schemas for the health endpoint.
"""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Response body returned by GET /api/v1/health."""

    status: str = Field(
        "healthy",
        description="Current operational status of the service",
        examples=["healthy"],
    )
    service: str = Field(
        ...,
        description="Name of the service",
        examples=["Fiduciary Lens Tax Advisor API"],
    )
    version: Optional[str] = Field(
        None,
        description="Current release version of the API",
        examples=["0.1.0"],
    )
    environment: Optional[str] = Field(
        None,
        description="Deployment environment (development, staging, production)",
        examples=["development"],
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "status": "healthy",
                "service": "Fiduciary Lens Tax Advisor API",
                "version": "0.1.0",
                "environment": "development",
            }
        }
    }


__all__ = ["HealthResponse"]
