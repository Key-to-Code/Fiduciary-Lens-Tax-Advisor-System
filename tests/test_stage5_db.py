"""
Stage 5 Persistent Storage Tests: PostgreSQL + SQLAlchemy + Alembic.

Validates:
1. PostgreSQL database connection and engine initialization.
2. Table structure and schema inspection.
3. Document persistence upon upload (filename, MIME type, file size, status).
4. Summary persistence upon summarization (linked foreign key, processing time).
5. Document history endpoints:
   - GET    /api/v1/documents
   - GET    /api/v1/documents/{document_id}
   - DELETE /api/v1/documents/{document_id} (cascade deletion of summaries)
6. Error handling (nonexistent document 404, failure status tracking without fake summaries).
7. OpenAPI schema validation including Stage 5 models.
"""

from __future__ import annotations

import sys
import uuid
from pathlib import Path

# Ensure project root is in sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import inspect, select, text

from backend.app.db.models import Document, Summary
from backend.app.db.session import SessionLocal, engine
from backend.app.main import app
from backend.app.schemas import (
    DocumentDeleteResponse,
    DocumentDetailResponse,
    DocumentListResponse,
    DocumentResponse,
    DocumentUploadResponse,
    ErrorResponse,
    SummarizeResponse,
)


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture
def db_session():
    """Provide a direct database session for test verification and cleanup."""
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


# ── 1. Database Connection & Schema Verification ─────────────────────────────

def test_postgresql_connection_and_engine():
    """Verify that SQLAlchemy connects to PostgreSQL and executes queries."""
    with engine.connect() as conn:
        result = conn.execute(text("SELECT 1;"))
        assert result.scalar() == 1


def test_database_tables_and_columns():
    """Verify that Alembic migrations created 'documents' and 'summaries' tables."""
    inspector = inspect(engine)
    tables = inspector.get_table_names()

    assert "documents" in tables
    assert "summaries" in tables

    doc_columns = {c["name"]: c for c in inspector.get_columns("documents")}
    for required_col in ("id", "filename", "file_type", "file_size", "processing_status", "created_at", "updated_at"):
        assert required_col in doc_columns, f"Missing column '{required_col}' in 'documents' table"

    sum_columns = {c["name"]: c for c in inspector.get_columns("summaries")}
    for required_col in ("id", "document_id", "summary", "processing_time", "created_at"):
        assert required_col in sum_columns, f"Missing column '{required_col}' in 'summaries' table"

    # Verify foreign key on summaries.document_id -> documents.id
    fks = inspector.get_foreign_keys("summaries")
    assert any(fk["referred_table"] == "documents" and "id" in fk["referred_columns"] for fk in fks)


# ── 2. Document Upload Persistence ───────────────────────────────────────────

def test_upload_document_creates_database_record(client: TestClient, db_session):
    """POST /api/v1/documents/upload persists document metadata in PostgreSQL."""
    content = b"Income-tax Act assessment provisions and Section 80C deductions for tax year 2026."
    filename = "stage5_test_upload.txt"

    response = client.post(
        "/api/v1/documents/upload",
        files={"file": (filename, content, "text/plain")},
    )
    assert response.status_code == 200
    data = response.json()
    assert "id" not in data
    doc_id = data["document_id"]

    # Verify response schema
    validated = DocumentUploadResponse.model_validate(data)
    assert validated.success is True
    assert validated.filename == filename

    # Verify persistence in database
    db_doc = db_session.get(Document, doc_id)
    assert db_doc is not None
    assert db_doc.id == doc_id
    assert db_doc.filename == filename
    assert db_doc.file_type == "text/plain"
    assert db_doc.file_size == len(content)
    assert db_doc.processing_status == "completed"
    assert db_doc.created_at is not None

    # Cleanup
    db_session.delete(db_doc)
    db_session.commit()


# ── 3. Summarization Persistence ──────────────────────────────────────────────

def test_summarize_file_creates_document_and_summary_records(client: TestClient, db_session):
    """POST /api/v1/summarize creates both Document and Summary database records."""
    text_content = (
        "FORM 16 SUMMARY: Gross total income INR 18,00,000. Standard deduction under Section 16(ia) "
        "is INR 75,000. Deductions under Section 80C: INR 1,50,000. Tax payable: INR 1,55,000."
    )
    filename = "stage5_tax_summary.txt"

    response = client.post(
        "/api/v1/summarize?provider=extractive",
        files={"file": (filename, text_content.encode("utf-8"), "text/plain")},
    )
    assert response.status_code == 200
    data = response.json()
    validated = SummarizeResponse.model_validate(data)
    doc_id = validated.document_id

    # Verify Document in DB
    db_doc = db_session.get(Document, doc_id)
    assert db_doc is not None
    assert db_doc.filename == filename
    assert db_doc.processing_status == "completed"

    # Verify Summary in DB
    stmt = select(Summary).where(Summary.document_id == doc_id)
    db_summaries = db_session.scalars(stmt).all()
    assert len(db_summaries) >= 1
    latest_summary = db_summaries[0]
    assert latest_summary.summary == validated.summary
    assert latest_summary.processing_time is not None
    assert latest_summary.processing_time >= 0.0

    # Cleanup
    db_session.delete(db_doc)
    db_session.commit()


def test_summarize_text_creates_persisted_records(client: TestClient, db_session):
    """POST /api/v1/summarize/text creates Document and Summary records."""
    payload = {
        "text": (
            "Tax Assessment Notice: Total income chargeable to tax is INR 9,50,000. "
            "Deductions claimed under Chapter VI-A are approved. Rebate under Section 87A applied."
        ),
        "provider": "extractive",
        "document_name": "direct_notice.txt",
    }
    response = client.post("/api/v1/summarize/text", json=payload)
    assert response.status_code == 200
    data = response.json()
    validated = SummarizeResponse.model_validate(data)
    doc_id = validated.document_id

    # Verify DB records
    db_doc = db_session.get(Document, doc_id)
    assert db_doc is not None
    assert db_doc.filename == "direct_notice.txt"
    assert db_doc.processing_status == "completed"

    stmt = select(Summary).where(Summary.document_id == doc_id)
    summaries = db_session.scalars(stmt).all()
    assert len(summaries) == 1
    assert summaries[0].summary == validated.summary

    # Cleanup
    db_session.delete(db_doc)
    db_session.commit()


# ── 4. Document History & Detail Endpoints ────────────────────────────────────

def test_list_documents_history(client: TestClient, db_session):
    """GET /api/v1/documents returns list of persisted documents."""
    doc_id = str(uuid.uuid4())
    doc = Document(
        id=doc_id,
        filename="history_sample.txt",
        file_type="text/plain",
        file_size=256,
        processing_status="completed",
    )
    db_session.add(doc)
    db_session.commit()

    response = client.get("/api/v1/documents")
    assert response.status_code == 200
    data = response.json()

    validated = DocumentListResponse.model_validate(data)
    assert validated.success is True
    assert validated.total >= 1
    matching = [d for d in validated.documents if d.document_id == doc_id]
    assert len(matching) == 1
    assert matching[0].filename == "history_sample.txt"
    for item in data["documents"]:
        assert "document_id" in item
        assert "id" not in item

    # Cleanup
    db_session.delete(doc)
    db_session.commit()


def test_get_document_detail_with_summary(client: TestClient, db_session):
    """GET /api/v1/documents/{id} returns document details with associated summaries."""
    doc_id = str(uuid.uuid4())
    doc = Document(
        id=doc_id,
        filename="detail_sample.txt",
        file_type="text/plain",
        file_size=512,
        processing_status="completed",
    )
    sum_id = str(uuid.uuid4())
    summary = Summary(
        id=sum_id,
        document_id=doc_id,
        summary="This is a test summary for detail endpoint verification.",
        processing_time=0.45,
    )
    db_session.add(doc)
    db_session.add(summary)
    db_session.commit()

    response = client.get(f"/api/v1/documents/{doc_id}")
    assert response.status_code == 200
    data = response.json()

    validated = DocumentDetailResponse.model_validate(data)
    assert validated.document_id == doc_id
    assert "id" not in data
    assert validated.filename == "detail_sample.txt"
    assert data["summaries"] is not None
    assert len(validated.summaries) == 1
    assert validated.summaries[0].id == sum_id
    assert validated.summaries[0].summary == "This is a test summary for detail endpoint verification."

    # Cleanup
    db_session.delete(doc)
    db_session.commit()


def test_get_nonexistent_document_returns_404(client: TestClient):
    """GET /api/v1/documents/{id} with invalid UUID returns 404 ErrorResponse."""
    nonexistent_id = str(uuid.uuid4())
    response = client.get(f"/api/v1/documents/{nonexistent_id}")
    assert response.status_code == 404
    data = response.json()

    validated = ErrorResponse.model_validate(data)
    assert validated.success is False
    assert validated.error.code == "DOCUMENT_NOT_FOUND"


# ── 5. Delete Endpoint & Cascade Deletion ─────────────────────────────────────

def test_delete_document_and_cascade_summaries(client: TestClient, db_session):
    """DELETE /api/v1/documents/{id} removes document and its cascaded summaries."""
    doc_id = str(uuid.uuid4())
    doc = Document(
        id=doc_id,
        filename="to_delete.txt",
        file_type="text/plain",
        file_size=128,
        processing_status="completed",
    )
    sum_id = str(uuid.uuid4())
    summary = Summary(
        id=sum_id,
        document_id=doc_id,
        summary="Summary that should be cascaded upon document deletion.",
        processing_time=0.12,
    )
    db_session.add(doc)
    db_session.add(summary)
    db_session.commit()

    # Verify both exist
    assert db_session.get(Document, doc_id) is not None
    assert db_session.get(Summary, sum_id) is not None

    # Delete via API
    response = client.delete(f"/api/v1/documents/{doc_id}")
    assert response.status_code == 200
    data = response.json()

    validated = DocumentDeleteResponse.model_validate(data)
    assert validated.success is True
    assert validated.document_id == doc_id

    # Verify document and summary are removed from PostgreSQL
    db_session.expire_all()
    assert db_session.get(Document, doc_id) is None
    assert db_session.get(Summary, sum_id) is None


def test_delete_nonexistent_document_returns_404(client: TestClient):
    """DELETE /api/v1/documents/{id} with missing ID returns 404 ErrorResponse."""
    nonexistent_id = str(uuid.uuid4())
    response = client.delete(f"/api/v1/documents/{nonexistent_id}")
    assert response.status_code == 404
    data = response.json()

    validated = ErrorResponse.model_validate(data)
    assert validated.success is False
    assert validated.error.code == "DOCUMENT_NOT_FOUND"


# ── 6. Error & Failure Handling ───────────────────────────────────────────────

def test_summarization_failure_updates_status_no_fake_summary(client: TestClient, db_session, monkeypatch):
    """When summarization raises an error, document status is 'failed' and no summary is created."""
    import backend.app.api.routes.summarize as summarize_mod
    from backend.app.services.rag_service import ProviderError

    def _failing_summarize(text: str, provider: str | None = None):
        raise ProviderError("Simulated LLM Failure", "Mocked failure for database status verification.")

    monkeypatch.setattr(summarize_mod, "summarize_legal_document", _failing_summarize)

    payload = {
        "text": "Valid document content that will trigger the mocked failure.",
        "document_name": "failing_doc.txt",
    }
    response = client.post("/api/v1/summarize/text", json=payload)
    assert response.status_code == 502
    data = response.json()

    validated = ErrorResponse.model_validate(data)
    assert validated.success is False
    assert validated.error.code == "PROVIDER_ERROR"

    # Verify that the document was marked as failed in DB
    stmt = select(Document).where(Document.filename == "failing_doc.txt").order_by(Document.created_at.desc())
    failed_doc = db_session.scalars(stmt).first()
    assert failed_doc is not None
    assert failed_doc.processing_status == "failed"

    # Verify NO fake summary was created
    sum_stmt = select(Summary).where(Summary.document_id == failed_doc.id)
    summaries = db_session.scalars(sum_stmt).all()
    assert len(summaries) == 0

    # Cleanup
    db_session.delete(failed_doc)
    db_session.commit()


# ── 7. OpenAPI Documentation Includes Stage 5 Schemas ─────────────────────────

def test_openapi_contains_stage5_schemas(client: TestClient):
    """OpenAPI schema contains all Stage 5 schemas."""
    response = client.get("/openapi.json")
    assert response.status_code == 200
    schema = response.json()

    components = schema.get("components", {}).get("schemas", {})
    expected_stage5_schemas = [
        "DocumentResponse",
        "DocumentDetailResponse",
        "DocumentListResponse",
        "DocumentDeleteResponse",
        "SummaryItemResponse",
    ]
    for name in expected_stage5_schemas:
        assert name in components, f"Missing Stage 5 schema '{name}' in OpenAPI documentation"

    document_props = components["DocumentResponse"].get("properties", {})
    assert "document_id" in document_props
    assert "id" not in document_props
    detail_props = components["DocumentDetailResponse"].get("properties", {})
    assert "document_id" in detail_props
    assert "id" not in detail_props
    assert "summaries" in detail_props
    summarize_body_schema = None
    for name, spec in components.items():
        if "summarize_document_endpoint" in name:
            summarize_body_schema = spec
            break
    assert summarize_body_schema is not None, "Missing OpenAPI body schema for POST /summarize"
    body_props = summarize_body_schema.get("properties", {})
    assert "file" in body_props
    assert "document_id" in body_props


def test_openapi_contains_stage5_paths(client: TestClient):
    """Live OpenAPI spec must advertise every Stage 5 history and summarize route."""
    response = client.get("/openapi.json")
    assert response.status_code == 200
    paths = response.json().get("paths", {})

    expected = {
        "/api/v1/documents": {"get"},
        "/api/v1/documents/{document_id}": {"get", "delete"},
        "/api/v1/documents/upload": {"post"},
        "/api/v1/summarize": {"post"},
        "/api/v1/summarize/text": {"post"},
        "/api/v1/health": {"get"},
    }
    for path, methods in expected.items():
        assert path in paths, f"Missing OpenAPI path '{path}'"
        advertised = {m.lower() for m in paths[path].keys() if m != "parameters"}
        for method in methods:
            assert method in advertised, f"Missing {method.upper()} {path} in OpenAPI"


def test_no_database_dependency_overrides():
    """The live app must use get_db() → SessionLocal → PostgreSQL, not test doubles."""
    assert app.dependency_overrides == {}


def test_engine_connects_to_legal_summarizer():
    """SQLAlchemy engine must target the legal_summarizer database."""
    with engine.connect() as conn:
        db_name = conn.execute(text("SELECT current_database()")).scalar()
    assert db_name == "legal_summarizer"


# ── 8. Document ID consistency & summary attachment ───────────────────────────

def _count_documents(db_session, doc_id: str | None = None) -> int:
    if doc_id is None:
        return db_session.execute(text("SELECT COUNT(*) FROM documents")).scalar_one()
    return db_session.execute(
        text("SELECT COUNT(*) FROM documents WHERE id = :id"),
        {"id": doc_id},
    ).scalar_one()


def test_upload_response_uses_canonical_document_id(client: TestClient, db_session):
    """TEST 1 — Upload returns document_id and no public id field."""
    content = b"Stage 5 identity test: Section 80C deductions and Form 16 salary details 2026."
    response = client.post(
        "/api/v1/documents/upload",
        files={"file": ("stage5_id_upload.txt", content, "text/plain")},
    )
    assert response.status_code == 200
    data = response.json()
    assert "document_id" in data
    assert "id" not in data
    DocumentUploadResponse.model_validate(data)

    db_session.expire_all()
    db_doc = db_session.get(Document, data["document_id"])
    assert db_doc is not None
    db_session.delete(db_doc)
    db_session.commit()


def test_summarize_reuses_existing_document_id(client: TestClient, db_session):
    """TEST 2/3 — Summarize with document_id does not create a second Document row."""
    content = (
        b"FORM 16: Gross salary INR 12,00,000. Section 80C deduction INR 1,50,000. "
        b"Tax payable INR 1,05,000. TDS deposited INR 1,05,000 for assessment year 2026."
    )
    upload = client.post(
        "/api/v1/documents/upload",
        files={"file": ("stage5_id_test.txt", content, "text/plain")},
    )
    assert upload.status_code == 200
    doc_id = upload.json()["document_id"]

    db_session.expire_all()
    count_before = _count_documents(db_session)
    count_for_id_before = _count_documents(db_session, doc_id)
    assert count_for_id_before == 1

    response = client.post(
        "/api/v1/summarize?provider=extractive",
        files={"file": ("stage5_id_test.txt", content, "text/plain")},
        data={"document_id": doc_id},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["document_id"] == doc_id

    db_session.expire_all()
    assert _count_documents(db_session, doc_id) == 1
    assert _count_documents(db_session) == count_before

    stmt = select(Summary).where(Summary.document_id == doc_id)
    summaries = db_session.scalars(stmt).all()
    assert len(summaries) >= 1
    assert summaries[0].document_id == doc_id
    assert summaries[0].summary
    assert data["summary"] == summaries[0].summary

    db_doc = db_session.get(Document, doc_id)
    db_session.delete(db_doc)
    db_session.commit()


def test_get_document_detail_returns_generated_summaries(client: TestClient, db_session):
    """TEST 4 — GET /documents/{id} returns summaries produced for that document."""
    content = (
        b"Tax notice: total income INR 9,50,000. Chapter VI-A deductions approved. "
        b"Rebate under Section 87A applied. Assessment year 2026-27."
    )
    upload = client.post(
        "/api/v1/documents/upload",
        files={"file": ("stage5_detail_summary.txt", content, "text/plain")},
    )
    assert upload.status_code == 200
    doc_id = upload.json()["document_id"]

    summarize = client.post(
        "/api/v1/summarize?provider=extractive",
        files={"file": ("stage5_detail_summary.txt", content, "text/plain")},
        data={"document_id": doc_id},
    )
    assert summarize.status_code == 200
    generated = summarize.json()["summary"]

    response = client.get(f"/api/v1/documents/{doc_id}")
    assert response.status_code == 200
    data = response.json()
    assert "id" not in data
    validated = DocumentDetailResponse.model_validate(data)
    assert validated.document_id == doc_id
    assert validated.summaries is not None
    assert len(validated.summaries) >= 1
    assert validated.summaries[0].document_id == doc_id
    assert generated in {item.summary for item in validated.summaries}
    assert all(item.summary for item in validated.summaries)

    db_session.expire_all()
    db_doc = db_session.get(Document, doc_id)
    db_session.delete(db_doc)
    db_session.commit()


def test_get_document_with_no_summary_returns_empty_list(client: TestClient, db_session):
    """TEST 5 — Uploaded document without summarization returns summaries=[]."""
    content = b"Unsummarized Form 16 text with enough characters for upload validation 2026."
    upload = client.post(
        "/api/v1/documents/upload",
        files={"file": ("stage5_no_summary.txt", content, "text/plain")},
    )
    assert upload.status_code == 200
    doc_id = upload.json()["document_id"]

    response = client.get(f"/api/v1/documents/{doc_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["summaries"] == []
    assert data["summaries"] is not None
    validated = DocumentDetailResponse.model_validate(data)
    assert validated.summaries == []

    db_session.expire_all()
    db_doc = db_session.get(Document, doc_id)
    db_session.delete(db_doc)
    db_session.commit()


def test_same_filename_receives_distinct_document_ids(client: TestClient, db_session):
    """TEST 6 — Identity is UUID-based; identical filenames are different documents."""
    content_a = b"First independent upload of Form 16 salary details for tax year 2026 AAA."
    content_b = b"Second independent upload of Form 16 salary details for tax year 2026 BBB."
    filename = "duplicate_name.txt"

    first = client.post(
        "/api/v1/documents/upload",
        files={"file": (filename, content_a, "text/plain")},
    )
    second = client.post(
        "/api/v1/documents/upload",
        files={"file": (filename, content_b, "text/plain")},
    )
    assert first.status_code == 200
    assert second.status_code == 200
    id_a = first.json()["document_id"]
    id_b = second.json()["document_id"]
    assert id_a != id_b

    db_session.expire_all()
    assert db_session.get(Document, id_a) is not None
    assert db_session.get(Document, id_b) is not None

    db_session.delete(db_session.get(Document, id_a))
    db_session.delete(db_session.get(Document, id_b))
    db_session.commit()


def test_document_list_uses_document_id_and_counts_total(client: TestClient, db_session):
    """TEST 7 — List endpoint exposes document_id, not a duplicate id field."""
    created_ids = []
    for idx in range(2):
        content = f"List history document {idx} with sufficient Form 16 tax content 2026.".encode()
        response = client.post(
            "/api/v1/documents/upload",
            files={"file": (f"stage5_list_{idx}.txt", content, "text/plain")},
        )
        assert response.status_code == 200
        created_ids.append(response.json()["document_id"])

    response = client.get("/api/v1/documents")
    assert response.status_code == 200
    data = response.json()
    validated = DocumentListResponse.model_validate(data)
    assert validated.total >= 2
    listed_ids = {item.document_id for item in validated.documents}
    assert set(created_ids).issubset(listed_ids)
    for raw in data["documents"]:
        assert "document_id" in raw
        assert "id" not in raw
        assert "summaries" not in raw

    db_session.expire_all()
    db_count = _count_documents(db_session)
    assert validated.total == db_count

    for doc_id in created_ids:
        db_session.delete(db_session.get(Document, doc_id))
    db_session.commit()


def test_delete_cascades_summaries_after_upload_summarize(client: TestClient, db_session):
    """TEST 8 — Delete removes the document row and cascaded summaries."""
    content = (
        b"Cascade delete Form 16: Gross salary INR 15,00,000. Section 80C INR 1,50,000. "
        b"Standard deduction INR 75,000. Tax year 2026."
    )
    upload = client.post(
        "/api/v1/documents/upload",
        files={"file": ("stage5_cascade.txt", content, "text/plain")},
    )
    doc_id = upload.json()["document_id"]
    summarize = client.post(
        "/api/v1/summarize?provider=extractive",
        files={"file": ("stage5_cascade.txt", content, "text/plain")},
        data={"document_id": doc_id},
    )
    assert summarize.status_code == 200

    db_session.expire_all()
    sum_ids = [
        row.id
        for row in db_session.scalars(select(Summary).where(Summary.document_id == doc_id)).all()
    ]
    assert len(sum_ids) >= 1

    response = client.delete(f"/api/v1/documents/{doc_id}")
    assert response.status_code == 200

    db_session.expire_all()
    assert db_session.get(Document, doc_id) is None
    remaining = db_session.scalars(select(Summary).where(Summary.document_id == doc_id)).all()
    assert remaining == []
    for sid in sum_ids:
        assert db_session.get(Summary, sid) is None


def test_summarize_unknown_document_id_returns_404(client: TestClient):
    """Supplying a non-existent document_id must not create a new document."""
    missing_id = str(uuid.uuid4())
    content = b"Enough legal tax text to pass extraction when document_id is missing in DB 2026."
    response = client.post(
        "/api/v1/summarize?provider=extractive",
        files={"file": ("missing_doc.txt", content, "text/plain")},
        data={"document_id": missing_id},
    )
    assert response.status_code == 404
    body = ErrorResponse.model_validate(response.json())
    assert body.error.code == "DOCUMENT_NOT_FOUND"
