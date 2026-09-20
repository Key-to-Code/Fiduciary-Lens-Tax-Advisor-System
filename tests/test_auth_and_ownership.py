"""Authentication and tenant-isolation regression tests against PostgreSQL."""

from __future__ import annotations

from datetime import timedelta
import uuid

from fastapi.testclient import TestClient

from backend.app.core.security import create_access_token
from backend.app.db.models import Document


def test_register_login_and_me_never_expose_password(client: TestClient) -> None:
    email = f"auth_{uuid.uuid4().hex[:12]}@example.com"
    password = "SecurePassword123"

    registered = client.post("/api/v1/auth/register", json={"email": email, "password": password})
    assert registered.status_code == 201
    assert "password" not in registered.json()

    duplicate = client.post("/api/v1/auth/register", json={"email": email, "password": password})
    assert duplicate.status_code == 409

    bad_login = client.post("/api/v1/auth/login", json={"email": email, "password": "wrong-password"})
    assert bad_login.status_code == 401

    login = client.post("/api/v1/auth/login", json={"email": email, "password": password})
    assert login.status_code == 200
    token = login.json()["access_token"]

    me = client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me.status_code == 200
    assert me.json()["email"] == email
    assert "password_hash" not in me.json()


def test_login_uses_json_email_password_and_validates_required_fields(client: TestClient) -> None:
    """The public login contract is JSON ``email`` + ``password``, never username."""
    password = "SecurePassword123"
    email = f"login_{uuid.uuid4().hex[:12]}@example.com"
    assert client.post("/api/v1/auth/register", json={"email": email, "password": password}).status_code == 201

    assert client.post("/api/v1/auth/login", json={"email": email, "password": password}).status_code == 200
    assert client.post("/api/v1/auth/login", json={"email": email}).status_code == 422
    assert client.post("/api/v1/auth/login", json={"password": password}).status_code == 422
    assert client.post(
        "/api/v1/auth/login", json={"email": "absent@example.com", "password": password}
    ).status_code == 401


def test_protected_routes_reject_missing_invalid_and_expired_tokens(client: TestClient, auth_user: dict) -> None:
    assert client.get("/api/v1/documents").status_code == 401
    assert client.get("/api/v1/documents", headers={"Authorization": "Bearer not-a-jwt"}).status_code == 401
    assert client.get("/api/v1/auth/me", headers={"Authorization": "Bearer not-a-jwt"}).status_code == 401
    assert client.get("/api/v1/auth/me").status_code == 401

    expired = create_access_token(auth_user["user_id"], expires_delta=timedelta(seconds=-1))
    response = client.get("/api/v1/documents", headers={"Authorization": f"Bearer {expired}"})
    assert response.status_code == 401
    assert client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {expired}"}).status_code == 401


def test_other_users_cannot_read_summarize_or_delete_documents(
    client: TestClient,
    db_session,
    auth_user: dict,
    auth_headers: dict,
) -> None:
    upload = client.post(
        "/api/v1/documents/upload",
        files={"file": ("owned.txt", b"Tax document owned only by the first test user in financial year 2026.", "text/plain")},
        headers=auth_headers,
    )
    assert upload.status_code == 200
    document_id = upload.json()["document_id"]

    other_email = f"other_{uuid.uuid4().hex[:12]}@example.com"
    password = "SecurePassword123"
    assert client.post("/api/v1/auth/register", json={"email": other_email, "password": password}).status_code == 201
    login = client.post("/api/v1/auth/login", json={"email": other_email, "password": password})
    other_headers = {"Authorization": f"Bearer {login.json()['access_token']}"}

    assert client.get(f"/api/v1/documents/{document_id}", headers=other_headers).status_code == 404
    assert client.delete(f"/api/v1/documents/{document_id}", headers=other_headers).status_code == 404
    summary_attempt = client.post(
        "/api/v1/summarize/text",
        json={"text": "Enough text to reach the ownership check without generating a summary.", "document_id": document_id},
        headers=other_headers,
    )
    assert summary_attempt.status_code == 404

    db_session.expire_all()
    document = db_session.get(Document, document_id)
    assert document is not None and document.user_id == auth_user["user_id"]
    db_session.delete(document)
    db_session.commit()


def test_openapi_advertises_bearer_security_and_auth_me(client: TestClient) -> None:
    schema = client.get("/openapi.json").json()
    assert "/api/v1/auth/me" in schema["paths"]
    login_body = schema["paths"]["/api/v1/auth/login"]["post"]["requestBody"]
    assert login_body["content"]["application/json"]["schema"]["$ref"].endswith("/LoginRequest")
    login_properties = schema["components"]["schemas"]["LoginRequest"]["properties"]
    assert set(login_properties) == {"email", "password"}
    schemes = schema["components"]["securitySchemes"]
    assert any(item.get("scheme") == "bearer" for item in schemes.values())
    assert schema["paths"]["/api/v1/documents"]["get"]["security"]
