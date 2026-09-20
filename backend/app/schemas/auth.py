"""
Pydantic schemas for registration, login, and safe user responses.

password and password_hash must never appear on UserResponse.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, EmailStr, Field


class RegisterRequest(BaseModel):
    email: EmailStr = Field(..., description="Unique account email address")
    password: str = Field(
        ...,
        min_length=8,
        max_length=72,
        description="Plaintext password (8–72 characters). Stored only as a bcrypt hash.",
    )


class LoginRequest(BaseModel):
    """JSON credentials accepted by ``POST /api/v1/auth/login``."""

    email: EmailStr = Field(..., description="Registered account email address")
    password: str = Field(
        ...,
        min_length=1,
        max_length=72,
        description="Account password. It is verified against the stored bcrypt hash.",
    )

    model_config = {
        "json_schema_extra": {
            "example": {"email": "userA@test.com", "password": "Test@12345"}
        }
    }


class UserResponse(BaseModel):
    user_id: str = Field(..., description="Public user identifier (UUID)")
    email: EmailStr
    created_at: datetime

    model_config = {
        "from_attributes": True,
        "json_schema_extra": {
            "example": {
                "user_id": "11111111-2222-3333-4444-555555555555",
                "email": "user@example.com",
                "created_at": "2026-09-20T16:00:00Z",
            }
        },
    }


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"

    model_config = {
        "json_schema_extra": {
            "example": {
                "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
                "token_type": "bearer",
            }
        },
    }
