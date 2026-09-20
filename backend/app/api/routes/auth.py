"""
Authentication routes.

POST /api/v1/auth/register  — create a user with a bcrypt password hash
POST /api/v1/auth/login     — validate JSON email/password; returns a JWT access token
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session

from backend.app.api.deps import get_current_user
from backend.app.core.security import create_access_token, hash_password, verify_password
from backend.app.db.models import User
from backend.app.db.session import get_db
from backend.app.schemas.auth import LoginRequest, RegisterRequest, TokenResponse, UserResponse
from backend.app.schemas.error import ErrorResponse

router = APIRouter(prefix="/auth", tags=["Authentication"])


def _normalize_email(email: str) -> str:
    return email.strip().lower()


@router.post(
    "/register",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
    responses={
        201: {"model": UserResponse, "description": "User registered"},
        409: {"model": ErrorResponse, "description": "Email already registered"},
        422: {"model": ErrorResponse, "description": "Invalid email or password"},
    },
    summary="Register a new user",
    description="Create an account. The password is hashed with bcrypt before it is stored.",
)
def register_user(payload: RegisterRequest, db: Session = Depends(get_db)) -> UserResponse:
    email = _normalize_email(str(payload.email))

    existing = db.scalars(select(User).where(User.email == email)).first()
    if existing is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "code": "EMAIL_ALREADY_REGISTERED",
                "message": "An account with this email already exists.",
                "details": None,
            },
        )

    user = User(
        email=email,
        password_hash=hash_password(payload.password),
    )
    try:
        db.add(user)
        db.commit()
        db.refresh(user)
    except IntegrityError:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "code": "EMAIL_ALREADY_REGISTERED",
                "message": "An account with this email already exists.",
                "details": None,
            },
        )
    except SQLAlchemyError:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "code": "DATABASE_ERROR",
                "message": "Database error",
                "details": "Failed to create user account.",
            },
        )

    return UserResponse(user_id=user.id, email=user.email, created_at=user.created_at)


@router.post(
    "/login",
    response_model=TokenResponse,
    responses={
        200: {"model": TokenResponse, "description": "JWT access token"},
        401: {"model": ErrorResponse, "description": "Invalid credentials"},
        422: {"model": ErrorResponse, "description": "Missing or invalid email/password request body"},
    },
    summary="Login",
    description=(
        "Submit a JSON request body with ``email`` and ``password``. "
        "Invalid email or password both return HTTP 401 without revealing which failed."
    ),
)
def login(
    payload: LoginRequest,
    db: Session = Depends(get_db),
) -> TokenResponse:
    email = _normalize_email(str(payload.email))
    user = db.scalars(select(User).where(User.email == email)).first()

    if user is None or not verify_password(payload.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "code": "UNAUTHORIZED",
                "message": "Invalid email or password.",
                "details": None,
            },
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = create_access_token(subject=user.id)
    return TokenResponse(access_token=token, token_type="bearer")


@router.get(
    "/me",
    response_model=UserResponse,
    responses={
        200: {"model": UserResponse, "description": "Authenticated user"},
        401: {"model": ErrorResponse, "description": "Missing, invalid, expired, or revoked JWT"},
    },
    summary="Get the current user",
    description="Return the account represented by the Bearer token. Password hashes are never exposed.",
)
def get_me(current_user: User = Depends(get_current_user)) -> UserResponse:
    """Return safe account metadata for the authenticated user."""
    return UserResponse(
        user_id=current_user.id,
        email=current_user.email,
        created_at=current_user.created_at,
    )
