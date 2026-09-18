"""
FastAPI application factory.

Entry point for uvicorn:
    uvicorn backend.app.main:app --reload

The module:
1. Ensures the project root is on sys.path so that 'shared', 'rag_model',
   and 'nlp_pipeline' remain importable exactly as they are from the CLI.
2. Configures CORS for the existing HTML front-end and common dev ports.
3. Mounts the v1 API router.
4. Exposes a redirect from / to /docs for convenience.
"""

from __future__ import annotations

import sys
from pathlib import Path

# ── sys.path bootstrap ───────────────────────────────────────────────────────
# When uvicorn is launched from backend/ or from the project root the import
# resolution for 'shared', 'rag_model', and 'nlp_pipeline' must work without
# any change to those packages. Inserting the project root once here mirrors
# the sys.path.insert(0, ...) found in the existing CLI scripts.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ── FastAPI & middleware imports ─────────────────────────────────────────────
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from backend.app.core.config import settings
from backend.app.api.routes import health as health_router
from backend.app.api.routes import documents as documents_router

# ── Application instance ─────────────────────────────────────────────────────
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=(
        "AI-powered Indian tax advisor. Powered by a hybrid RAG engine over the "
        "Income-tax Act 2025 (as amended by the Finance Act 2026) and an NLP "
        "pipeline for tax document analysis.\n\n"
        "**Disclaimer**: Educational information only, not professional tax, "
        "accounting or financial advice."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# ── CORS ─────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
# All routes are versioned under /api/v1 for clean forward-compatibility.
API_PREFIX = "/api/v1"

app.include_router(health_router.router, prefix=API_PREFIX)
app.include_router(documents_router.router, prefix=API_PREFIX)

# ── Convenience redirect ──────────────────────────────────────────────────────
@app.get("/", include_in_schema=False)
async def root_redirect() -> RedirectResponse:
    """Redirect bare root to the interactive API docs."""
    return RedirectResponse(url="/docs")


# ── Startup / shutdown events ─────────────────────────────────────────────────
@app.on_event("startup")
async def on_startup() -> None:
    """
    Startup hook — lightweight validation only.

    Heavy pipeline components (FAISS index, embedding model, LLM client) are
    intentionally NOT loaded here. They initialise on first request so the
    server starts instantly and the health endpoint is always fast.
    """
    print(f"[startup] {settings.APP_NAME} v{settings.APP_VERSION}")
    print(f"[startup] environment : {settings.APP_ENV}")
    print(f"[startup] project root: {_PROJECT_ROOT}")
    print(f"[startup] CORS origins: {settings.CORS_ORIGINS}")
    print("[startup] API docs available at /docs")


@app.on_event("shutdown")
async def on_shutdown() -> None:
    print(f"[shutdown] {settings.APP_NAME} shutting down.")
