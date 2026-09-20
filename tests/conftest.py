"""
Shared pytest fixtures for API tests.

Stage 6: authenticated requests use a freshly registered user and Bearer JWT.
Health and OpenAPI remain public.
"""

from __future__ import annotations

import sys
import uuid
import os
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# The application deliberately requires a JWT secret from configuration. This
# non-production value exists only in the test process, before settings import.
os.environ.setdefault("JWT_SECRET_KEY", "test-only-jwt-secret-that-is-at-least-32-chars")

import pytest
from fastapi.testclient import TestClient

from backend.app.db.session import SessionLocal
from backend.app.main import app


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture
def db_session():
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def auth_user(client: TestClient) -> dict:
    email = f"pytest_{uuid.uuid4().hex[:12]}@example.com"
    password = "SecurePassword123"
    register = client.post("/api/v1/auth/register", json={"email": email, "password": password})
    assert register.status_code == 201, register.text
    body = register.json()
    login = client.post(
        "/api/v1/auth/login",
        json={"email": email, "password": password},
    )
    assert login.status_code == 200, login.text
    token = login.json()["access_token"]
    return {
        "email": email,
        "password": password,
        "user_id": body["user_id"],
        "token": token,
        "headers": {"Authorization": f"Bearer {token}"},
    }


@pytest.fixture
def auth_headers(auth_user: dict) -> dict:
    return auth_user["headers"]
