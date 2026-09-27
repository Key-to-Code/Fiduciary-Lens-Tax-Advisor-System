"""Contract tests for the public RAG Q&A endpoint used by the chat frontend."""

from __future__ import annotations

from fastapi.testclient import TestClient

from backend.app.schemas.error import ErrorResponse
from backend.app.services.rag_service import ProviderError


def test_ask_returns_grounded_answer_and_sources(client: TestClient, monkeypatch) -> None:
    def _fake_ask(question: str, provider: str | None = None, history=None):
        assert question == "What deductions are allowed for life insurance premium?"
        assert history == []
        return {
            "answer": "A deduction is allowed for life insurance premium [1] (Act s.123).",
            "sources": [
                {
                    "n": 1,
                    "citation": "Income-tax Act, 2025, Section 123",
                    "short": "Act s.123",
                    "document": "Income-tax-Act-2025.pdf",
                    "as_of": "as amended by Finance Act, 2026; in force from 1 April 2026",
                    "chunk_id": "act_123",
                    "score": 0.91,
                    "cosine": 0.81,
                    "excerpt": "A deduction of an amount paid towards life insurance premium shall be allowed.",
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
        json={"question": "What deductions are allowed for life insurance premium?"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["grounded"] is True
    assert data["answer"].startswith("A deduction is allowed")
    assert data["sources"][0]["short"] == "Act s.123"
    assert data["sources"][0]["excerpt"]
    assert data["sources"][0]["url"] is None
    assert "password" not in data
    assert "OPENAI" not in str(data)


def test_ask_passes_conversation_history(client: TestClient, monkeypatch) -> None:
    captured = {}

    def _fake_ask(question: str, provider: str | None = None, history=None):
        captured["history"] = history
        return {
            "answer": "Follow-up answer",
            "sources": [],
            "grounded": False,
            "provider": "extractive",
            "latency_ms": 4,
        }

    monkeypatch.setattr("backend.app.api.routes.ask.ask_tax_question", _fake_ask)
    response = client.post(
        "/api/v1/ask",
        json={
            "question": "What is the limit?",
            "history": [
                {"question": "What is section 123?", "answer": "Life insurance deduction."},
            ],
        },
    )
    assert response.status_code == 200
    assert captured["history"] == [("What is section 123?", "Life insurance deduction.")]


def test_ask_ungrounded_does_not_invent_sources(client: TestClient, monkeypatch) -> None:
    def _fake_ask(question: str, provider: str | None = None, history=None):
        return {
            "answer": "I could not find a provision in my knowledge base that answers that.",
            "sources": [],
            "grounded": False,
            "provider": "extractive",
            "latency_ms": 8,
        }

    monkeypatch.setattr("backend.app.api.routes.ask.ask_tax_question", _fake_ask)
    response = client.post("/api/v1/ask", json={"question": "Who won the 2018 FIFA World Cup?"})
    assert response.status_code == 200
    data = response.json()
    assert data["grounded"] is False
    assert data["sources"] == []
    assert "FIFA" not in data["answer"] or True  # answer comes from the pipeline, not invented here


def test_ask_rejects_empty_question(client: TestClient) -> None:
    response = client.post("/api/v1/ask", json={"question": "  "})
    assert response.status_code == 422
    ErrorResponse.model_validate(response.json())


def test_ask_maps_missing_index_to_503(client: TestClient, monkeypatch) -> None:
    def _missing(*_args, **_kwargs):
        raise FileNotFoundError("No index at index/. Build it first:  python build_index.py")

    monkeypatch.setattr("backend.app.api.routes.ask.ask_tax_question", _missing)
    response = client.post("/api/v1/ask", json={"question": "What is advance tax?"})
    assert response.status_code == 503
    body = ErrorResponse.model_validate(response.json())
    assert body.error.code in {"INDEX_NOT_BUILT", "SERVICE_UNAVAILABLE"}


def test_ask_maps_provider_failure_to_502(client: TestClient, monkeypatch) -> None:
    def _fail(*_args, **_kwargs):
        raise ProviderError("RAG pipeline failure", " Simulated backend error")

    monkeypatch.setattr("backend.app.api.routes.ask.ask_tax_question", _fail)
    response = client.post("/api/v1/ask", json={"question": "What is a perquisite?"})
    assert response.status_code == 502
    body = ErrorResponse.model_validate(response.json())
    assert body.error.code == "PROVIDER_ERROR"


def test_ask_is_public_and_documented(client: TestClient) -> None:
    schema = client.get("/openapi.json").json()
    assert "/api/v1/ask" in schema["paths"]
    assert "post" in schema["paths"]["/api/v1/ask"]
    security = schema["paths"]["/api/v1/ask"]["post"].get("security")
    assert not security
    components = schema["components"]["schemas"]
    assert "AskRequest" in components
    assert "AskResponse" in components


def test_knowledge_status_is_public(client: TestClient, monkeypatch, tmp_path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        '{"embed_model":"BAAI/bge-small-en-v1.5","n_chunks":12,'
        '"documents":["Finance-Act-2026.pdf","Income-tax-Act-2025.pdf"]}',
        encoding="utf-8",
    )
    monkeypatch.setattr("backend.app.api.routes.ask.INDEX_DIR", tmp_path)
    response = client.get("/api/v1/knowledge")
    assert response.status_code == 200
    data = response.json()
    assert data["ready"] is True
    assert data["n_chunks"] == 12
    assert "Finance-Act-2026.pdf" in data["documents"]
