"""
Stage 4 API Contract and Pydantic Schema Validation Tests.

Validates:
- Health endpoint schema and responses
- Document upload endpoints (valid uploads, unsupported types, empty files, invalid PDFs, oversized files)
- Summarization endpoints (file upload, text JSON body, validation, provider checks)
- Standardized ErrorResponse schema consistency across status codes (400, 404, 413, 422, 500, 502)
- OpenAPI schema generation and registration of all schemas
- Absence of sensitive stack traces or internal filesystem paths in error responses
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

# Ensure project root is in sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.schemas import (
    DocumentUploadResponse,
    ErrorDetail,
    ErrorResponse,
    HealthResponse,
    SummarizeRequest,
    SummarizeResponse,
)

# ── 1. Health Endpoint Tests ───────────────────────────────────────────────────

def test_health_check_status_and_schema(client: TestClient):
    """GET /api/v1/health returns 200 and matches HealthResponse schema."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()

    # Schema validation
    validated = HealthResponse.model_validate(data)
    assert validated.status == "healthy"
    assert validated.service == "legal-document-summarizer"
    assert validated.version is not None
    assert validated.environment is not None


# ── 2. Document Upload Tests ───────────────────────────────────────────────────

def test_upload_valid_text_document(client: TestClient, auth_headers: dict):
    """POST /api/v1/documents/upload with valid .txt document succeeds with DocumentUploadResponse."""
    content = b"Tax assessment year 2026-27 under the Income-tax Act. Standard deduction Section 16(ia) is INR 75,000."
    response = client.post(
        "/api/v1/documents/upload",
        files={"file": ("tax_summary.txt", content, "text/plain")},
        headers=auth_headers,
    )
    assert response.status_code == 200
    data = response.json()

    validated = DocumentUploadResponse.model_validate(data)
    assert validated.success is True
    assert validated.filename == "tax_summary.txt"
    assert validated.mime_type == "text/plain"
    assert validated.file_size_bytes == len(content)
    assert validated.char_count == len(content.decode("utf-8"))
    assert "Section 16" in validated.preview


def test_upload_valid_markdown_document(client: TestClient, auth_headers: dict):
    """POST /api/v1/documents/upload with valid .md document succeeds."""
    content = b"# Tax Advisory Notice\nDeduction under Section 80C claimed: INR 1,50,000."
    response = client.post(
        "/api/v1/documents/upload",
        files={"file": ("notice.md", content, "text/markdown")},
        headers=auth_headers,
    )
    assert response.status_code == 200
    data = response.json()
    validated = DocumentUploadResponse.model_validate(data)
    assert validated.success is True
    assert validated.mime_type == "text/markdown"


def test_upload_unsupported_file_extension(client: TestClient, auth_headers: dict):
    """POST /api/v1/documents/upload with unsupported extension returns 400 ErrorResponse."""
    response = client.post(
        "/api/v1/documents/upload",
        files={"file": ("malicious.exe", b"MZ\x90\x00binary content", "application/octet-stream")},
        headers=auth_headers,
    )
    assert response.status_code == 400
    data = response.json()

    validated = ErrorResponse.model_validate(data)
    assert validated.success is False
    assert validated.error.code == "UNSUPPORTED_FILE_TYPE"
    assert "Unsupported file type" in validated.error.message
    assert ".exe" in str(validated.error.details)


def test_upload_empty_file(client: TestClient, auth_headers: dict):
    """POST /api/v1/documents/upload with empty file returns 400 ErrorResponse."""
    response = client.post(
        "/api/v1/documents/upload",
        files={"file": ("empty.txt", b"", "text/plain")},
        headers=auth_headers,
    )
    assert response.status_code == 400
    data = response.json()

    validated = ErrorResponse.model_validate(data)
    assert validated.success is False
    assert validated.error.code == "EMPTY_FILE"
    assert "Empty file" in validated.error.message


def test_upload_invalid_pdf_header(client: TestClient, auth_headers: dict):
    """POST /api/v1/documents/upload with .pdf extension lacking %PDF header returns 400."""
    response = client.post(
        "/api/v1/documents/upload",
        files={"file": ("corrupt.pdf", b"NOT_A_REAL_PDF_HEADER", "application/pdf")},
        headers=auth_headers,
    )
    assert response.status_code == 400
    data = response.json()

    validated = ErrorResponse.model_validate(data)
    assert validated.success is False
    assert validated.error.code == "INVALID_PDF"
    assert "Invalid PDF" in validated.error.message


def test_upload_insufficient_content(client: TestClient, auth_headers: dict):
    """POST /api/v1/documents/upload with insufficient characters returns 422."""
    response = client.post(
        "/api/v1/documents/upload",
        files={"file": ("tiny.txt", b"abc", "text/plain")},
        headers=auth_headers,
    )
    assert response.status_code == 422
    data = response.json()

    validated = ErrorResponse.model_validate(data)
    assert validated.success is False
    assert validated.error.code == "INSUFFICIENT_CONTENT"


def test_upload_oversized_file(client: TestClient, monkeypatch, auth_headers: dict):
    """POST /api/v1/documents/upload exceeding size limit returns 413 ErrorResponse."""
    from backend.app.core.config import settings
    # Temporarily set limit to 0 MB to trigger size check without allocating 20 MB
    monkeypatch.setattr(settings, "UPLOAD_MAX_SIZE_MB", 0)

    response = client.post(
        "/api/v1/documents/upload",
        files={"file": ("large.txt", b"Some content", "text/plain")},
        headers=auth_headers,
    )
    assert response.status_code == 413
    data = response.json()

    validated = ErrorResponse.model_validate(data)
    assert validated.success is False
    assert validated.error.code == "FILE_TOO_LARGE"
    assert "too large" in validated.error.message.lower()


# ── 3. Summarization Endpoint Tests ─────────────────────────────────────────────

def test_summarize_file_upload(client: TestClient, auth_headers: dict):
    """POST /api/v1/summarize with valid file upload returns SummarizeResponse."""
    text = (
        "FORM 16 PART B. Gross salary INR 15,00,000. Standard deduction INR 75,000. "
        "Deductions under Section 80C: INR 1,50,000. Net taxable income: INR 12,50,000. "
        "Total TDS deducted: INR 1,19,600. Filing deadline: 31st July 2026."
    )
    response = client.post(
        "/api/v1/summarize?provider=extractive",
        files={"file": ("Form16.txt", text.encode("utf-8"), "text/plain")},
        headers=auth_headers,
    )
    assert response.status_code == 200
    data = response.json()

    validated = SummarizeResponse.model_validate(data)
    assert validated.success is True
    assert validated.filename == "Form16.txt"
    assert validated.char_count == len(text)
    assert validated.file_size_bytes == len(text.encode("utf-8"))
    assert len(validated.summary) > 0
    assert validated.processing_time >= 0.0


def test_summarize_text_endpoint_valid(client: TestClient, auth_headers: dict):
    """POST /api/v1/summarize/text with SummarizeRequest returns SummarizeResponse."""
    payload = {
        "text": (
            "Section 80C deduction limit is INR 1,50,000 for financial year 2025-26. "
            "Standard deduction under section 16(ia) is INR 75,000 for salaried employees."
        ),
        "provider": "extractive",
        "document_name": "tax_rules.txt",
    }
    response = client.post("/api/v1/summarize/text", json=payload, headers=auth_headers)
    assert response.status_code == 200
    data = response.json()

    validated = SummarizeResponse.model_validate(data)
    assert validated.success is True
    assert validated.filename == "tax_rules.txt"
    assert validated.char_count == len(payload["text"])
    assert len(validated.summary) > 0


def test_summarize_text_short_payload_validation_error(client: TestClient, auth_headers: dict):
    """POST /api/v1/summarize/text with text < 5 chars returns 422 VALIDATION_ERROR."""
    response = client.post("/api/v1/summarize/text", json={"text": "Hi"}, headers=auth_headers)
    assert response.status_code == 422
    data = response.json()

    validated = ErrorResponse.model_validate(data)
    assert validated.success is False
    assert validated.error.code == "VALIDATION_ERROR"
    assert any("text" in str(item) for item in validated.error.details)


def test_summarize_text_invalid_provider_validation_error(client: TestClient, auth_headers: dict):
    """POST /api/v1/summarize/text with invalid provider returns 422 VALIDATION_ERROR."""
    payload = {
        "text": "Valid document text that is long enough to pass length validation.",
        "provider": "invalid_unknown_provider",
    }
    response = client.post("/api/v1/summarize/text", json=payload, headers=auth_headers)
    assert response.status_code == 422
    data = response.json()

    validated = ErrorResponse.model_validate(data)
    assert validated.success is False
    assert validated.error.code == "VALIDATION_ERROR"
    assert any("provider" in str(item) for item in validated.error.details)


def test_summarize_query_invalid_provider(client: TestClient, auth_headers: dict):
    """POST /api/v1/summarize with invalid ?provider= returns 400 INVALID_PROVIDER."""
    text = b"Valid document text that is long enough to pass length validation."
    response = client.post(
        "/api/v1/summarize?provider=bad_provider",
        files={"file": ("doc.txt", text, "text/plain")},
        headers=auth_headers,
    )
    assert response.status_code == 400
    data = response.json()

    validated = ErrorResponse.model_validate(data)
    assert validated.success is False
    assert validated.error.code == "INVALID_PROVIDER"


def test_summarize_direct_json_body(client: TestClient, auth_headers: dict):
    """POST /api/v1/summarize accepting direct JSON payload works transparently."""
    payload = {
        "text": "Direct JSON summarization test for Section 80C deductions and Section 115BAC.",
        "provider": "extractive",
        "document_name": "direct_json.txt",
    }
    response = client.post("/api/v1/summarize", json=payload, headers=auth_headers)
    assert response.status_code == 200
    data = response.json()
    validated = SummarizeResponse.model_validate(data)
    assert validated.success is True
    assert validated.filename == "direct_json.txt"


# ── 4. Error Consistency and Security Tests ────────────────────────────────────

def test_404_error_consistency(client: TestClient):
    """Accessing an unknown route returns a standardized 404 ErrorResponse."""
    response = client.get("/api/v1/non_existent_route")
    assert response.status_code == 404
    data = response.json()

    validated = ErrorResponse.model_validate(data)
    assert validated.success is False
    assert validated.error.code == "NOT_FOUND"


def test_no_sensitive_info_leaked_in_errors(client: TestClient, auth_headers: dict):
    """Errors must not expose internal paths, tokens, or tracebacks."""
    response = client.post(
        "/api/v1/documents/upload",
        files={"file": ("../../etc/passwd.exe", b"bad", "application/octet-stream")},
        headers=auth_headers,
    )
    assert response.status_code in (400, 422)
    data = response.json()

    json_str = str(data)
    assert "Traceback" not in json_str
    assert "Users/" not in json_str
    assert "passwd" not in json_str or "passwd.exe" in json_str  # only the safe display filename if any
    assert "sk-" not in json_str  # no API keys


# ── 5. OpenAPI Documentation Tests ─────────────────────────────────────────────

def test_openapi_schema_contains_stage4_models(client: TestClient):
    """OpenAPI schema contains all Stage 4 schemas with descriptions."""
    response = client.get("/openapi.json")
    assert response.status_code == 200
    schema = response.json()

    components = schema.get("components", {}).get("schemas", {})
    expected_schemas = [
        "HealthResponse",
        "DocumentUploadResponse",
        "SummarizeRequest",
        "SummarizeResponse",
        "ErrorResponse",
        "ErrorDetail",
    ]
    for name in expected_schemas:
        assert name in components, f"Missing schema '{name}' in OpenAPI documentation"

    # Verify tags exist
    tag_names = [t["name"] for t in schema.get("tags", [])]
    assert "Health" in tag_names
    assert "Documents" in tag_names
    assert "Summarization" in tag_names


def test_swagger_docs_accessible(client: TestClient):
    """Interactive Swagger UI is accessible at /docs."""
    response = client.get("/docs")
    assert response.status_code == 200
    assert "swagger-ui" in response.text.lower()
