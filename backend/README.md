# Fiduciary Lens — FastAPI Backend

Stage 1 backend foundation for the Fiduciary Lens Tax Advisor System.

---

## Structure

```
backend/
├── app/
│   ├── main.py              # FastAPI app, CORS, router mounting, startup hook
│   ├── core/
│   │   └── config.py        # pydantic-settings: reads root .env
│   ├── api/
│   │   └── routes/
│   │       └── health.py    # GET /api/v1/health
│   ├── services/
│   │   └── rag_service.py   # Integration stub for TaxQA + generate_insights
│   ├── schemas/
│   │   └── health.py        # HealthResponse Pydantic model
│   └── utils/               # (reserved for future helpers)
├── requirements.txt          # Web-layer deps only (FastAPI, uvicorn, pydantic-settings)
├── .env.example              # All env vars documented; copy to project root as .env
└── README.md                 # This file
```

---

## Prerequisites

1. **Install all dependencies** (from the project root):

   ```bash
   pip install -r requirements.txt
   pip install -r backend/requirements.txt
   ```

2. **Configure environment** — copy the example and fill in your credentials:

   ```bash
   cp backend/.env.example .env   # .env lives at the project root
   ```

   Minimum required values for the RAG pipeline:
   ```env
   OPENAI_API_KEY=your_key_here
   OPENAI_BASE_URL=https://openrouter.ai/api/v1   # or omit for api.openai.com
   HF_TOKEN=your_hf_token_here
   ```

3. **Build the FAISS index** (only needed to call RAG endpoints; not required for the health check):

   ```bash
   python rag_model/scripts/build_index.py
   ```

---

## Running the Backend

Run from the **project root** so that `shared/`, `rag_model/`, and `nlp_pipeline/` are importable:

```bash
# Development (auto-reload on file changes)
uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000

# Production
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --workers 1
```

---

## Endpoints (Stage 1)

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/v1/health` | Liveness probe — always fast, no I/O |
| GET | `/docs` | Swagger UI |
| GET | `/redoc` | ReDoc UI |
| GET | `/` | Redirects to `/docs` |

### Health check

```bash
curl http://localhost:8000/api/v1/health
```

```json
{"status": "healthy", "service": "legal-document-summarizer"}
```

---

## Design Notes

- **sys.path bootstrap** — `main.py` inserts the project root into `sys.path` using the same pattern as the existing CLI scripts, so `shared`, `rag_model`, and `nlp_pipeline` are importable without any modification.
- **Deferred pipeline init** — The FAISS index and embedding models are **not** loaded at server startup. They initialise on the first RAG request, keeping startup instant.
- **No secret hardcoding** — all credentials come from the `.env` file; the backend layer adds no new secrets.
- **Single source of truth** — `shared/config.py` continues to own all RAG/NLP settings. `backend/app/core/config.py` only manages web-layer settings (CORS, host, port).

---

## Existing CLI — Unchanged

The existing CLI workflows continue to work exactly as documented in the root `README.md`:

```bash
python rag_model/scripts/ask.py "What deductions are allowed under Section 80C?"
python nlp_pipeline/src/insight.py "rag_model/data/synthetic_forms/Form16_Synthetic_Arjun_Sharma.pdf"
```
