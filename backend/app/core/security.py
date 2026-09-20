"""
Password hashing and JWT helpers.

Passwords are hashed with bcrypt. JWTs are signed with HS256 using
JWT_SECRET_KEY from the existing Settings object.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional

import bcrypt
import jwt
from jwt.exceptions import ExpiredSignatureError, InvalidTokenError, PyJWTError

from backend.app.core.config import settings


class TokenError(Exception):
    """Raised when a JWT is missing, malformed, expired, or otherwise invalid."""


def hash_password(password: str) -> str:
    """Return a bcrypt hash for the given plaintext password."""
    password_bytes = password.encode("utf-8")
    return bcrypt.hashpw(password_bytes, bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    """Return True if plaintext matches the stored bcrypt hash."""
    try:
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
    except (ValueError, TypeError):
        return False


def create_access_token(subject: str, expires_delta: Optional[timedelta] = None) -> str:
    """
    Create a signed JWT whose ``sub`` claim is the stable user id.

    ``expires_delta`` is intended for tests (e.g. already-expired tokens).
    """
    if expires_delta is None:
        expires_delta = timedelta(minutes=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES)
    expire = datetime.now(timezone.utc) + expires_delta
    payload = {"sub": subject, "exp": expire}
    return jwt.encode(payload, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)


def decode_access_token(token: str) -> str:
    """
    Validate signature, algorithm, and expiration. Return the ``sub`` user id.

    Raises TokenError for any validation failure. Callers must not distinguish
    "user not found" vs "bad password" at the HTTP layer using this exception.
    """
    try:
        payload = jwt.decode(
            token,
            settings.JWT_SECRET_KEY,
            algorithms=[settings.JWT_ALGORITHM],
            options={"require": ["exp", "sub"]},
        )
    except ExpiredSignatureError as exc:
        raise TokenError("expired") from exc
    except (InvalidTokenError, PyJWTError, ValueError, TypeError) as exc:
        raise TokenError("invalid") from exc

    subject = payload.get("sub")
    if not subject or not isinstance(subject, str):
        raise TokenError("invalid")
    return subject
