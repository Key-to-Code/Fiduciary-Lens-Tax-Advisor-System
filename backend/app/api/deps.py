"""
Reusable FastAPI dependencies for Stage 6 authentication.
"""

from __future__ import annotations

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from backend.app.core.security import TokenError, decode_access_token
from backend.app.db.models import User
from backend.app.db.session import get_db

# auto_error=False so missing credentials become HTTP 401 (HTTPBearer defaults to 403).
bearer_scheme = HTTPBearer(auto_error=False)


def _unauthorized(message: str = "Could not validate credentials") -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail={"code": "UNAUTHORIZED", "message": message, "details": None},
        headers={"WWW-Authenticate": "Bearer"},
    )


def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    db: Session = Depends(get_db),
) -> User:
    """
    Extract a Bearer JWT, validate it, and load the User from PostgreSQL.

    Invalid, expired, malformed, or missing tokens return HTTP 401.
    A valid token whose subject no longer exists also returns HTTP 401.
    """
    if credentials is None or credentials.scheme.lower() != "bearer" or not credentials.credentials:
        raise _unauthorized("Not authenticated")

    try:
        user_id = decode_access_token(credentials.credentials)
    except TokenError:
        raise _unauthorized() from None

    user = db.get(User, user_id)
    if user is None:
        raise _unauthorized()
    return user
