"""Public RAG Q&A endpoint used by the chat frontend."""

from __future__ import annotations

from fastapi.testclient import TestClient

from backend.app.schemas.error import ErrorResponse
from backend.app.services.rag_service import ProviderError


def test_ask_is_public_and_returns_grounded_payload(client: TestClient, monkeypatch) -> None:
    def _fake_ask(question: str, provider: str | None = None, history=None):
        assert question == "What deductions are allowed under Section 80C?"
        assert history == []
        return {
            "answer": "A deduction is allowed under [1] (Act s.123).",
            "sources": [
                {
                    "n": 1,
                    "citation": "Income-tax Act, 2025, Section 123",
                    "short": "Act s.123",
                    "document": "Income-tax-Act-2025.pdf",
                    "as_of": "as amended by Finance Act, 2026",
                    "chunk_id": "act_chunk_1",
                    "score": 0.91,
                    "cosine": 0.81,
                    "excerpt": "A deduction of one hundred and fifty thousand rupees shall be allowed.",
                    "url": None,
                }
            ],
            "grounded": True,
            "provider": "extractive",
            "latency_ms": 12,
        }

    monkeypatch.setattr("backend.app.api.routes.ask.ask_tax_question", _fake_ask)

    response = client.post(
        "/api/v1/ask",
        json={"question": "What deductions are allowed under Section 80C?"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["grounded"] is True
    assert data["answer"].startswith("A deduction")
    assert data["sources"][0]["short"] == "Act s.123"
    assert data["sources"][0]["excerpt"]
    assert data["sources"][0]["url"] is None
    assert "password" not in str(data).lower()
    assert "api_key" not in str(data).lower()


def test_ask_ungrounded_returns_empty_sources(client: TestClient, monkeypatch) -> None:
    monkeypatch.setattr(
        "backend.app.api.routes.ask.ask_tax_question",
        lambda question, provider=None, history=None: {
            "answer": "I could not find a provision in my knowledge base that answers that.",
            "sources": [],
            "grounded": False,
            "provider": "extractive",
            "latency_ms": 8,
        },
    )
    response = client.post("/api/v1/ask", json={"question": "Who won the 2018 FIFA World Cup?"})
    assert response.status_code == 200
    data = response.json()
    assert data["grounded"] is False
    assert data["sources"] == []
    assert data["answer"]


def test_ask_empty_question_is_validation_error(client: TestClient) -> None:
    response = client.post("/api/v1/ask", json={"question": "  "})
    assert response.status_code == 422
    ErrorResponse.model_validate(response.json())


def test_ask_provider_error_is_502(client: TestClient, monkeypatch) -> None:
    def _fail(question: str, provider: str | None = None, history=None):
        raise ProviderError("RAG pipeline failure", "index missing")

    monkeypatch.setattr("backend.app.api.routes.ask.ask_tax_question", _fail)
    response = client.post("/api/v1/ask", json={"question": "What is a perquisite?"})
    assert response.status_code == 502
    body = ErrorResponse.model_validate(response.json())
    assert body.error.code == "PROVIDER_ERROR"


def test_knowledge_status_is_public(client: TestClient, monkeypatch, tmp_path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        '{"embed_model":"BAAI/bge-small-en-v1.5","n_chunks":12,'
        '"documents":["Finance-Act-2026.pdf","Income-tax-Act-2025.pdf"]}',
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "backend.app.api.routes.ask.config.INDEX_DIR",
        tmp_path,
    )
    response = client.get("/api/v1/knowledge")
    assert response.status_code == 200
    data = response.json()
    assert data["available"] is True
    assert data["n_chunks"] == 12
    assert "Finance-Act-2026.pdf" in data["documents"]


def test_openapi_includes_ask_route(client: TestClient) -> None:
    schema = client.get("/openapi.json").json()
    assert "/api/v1/ask" in schema["paths"]
    assert "post" in schema["paths"]["/api/v1/ask"]
    assert "/api/v1/knowledge" in schema["paths"]
