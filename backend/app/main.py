"""
FastAPI application factory.

Entry point for uvicorn:
    uvicorn backend.app.main:app --reload

The module:
1. Ensures the project root is on sys.path so that 'shared', 'rag_model',
   and 'nlp_pipeline' remain importable exactly as they are from the CLI.
2. Configures CORS for the existing HTML front-end and common dev ports.
3. Registers centralized exception handlers standardizing all 4xx/5xx responses.
4. Mounts the v1 API router.
5. Exposes a redirect from / to /docs for convenience.
"""

from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from pathlib import Path

# ── sys.path bootstrap ───────────────────────────────────────────────────────
# When uvicorn is launched from backend/ or from the project root the import
# resolution for 'shared', 'rag_model', and 'nlp_pipeline' must work without
# any change to those packages. Inserting the project root once here mirrors
# the sys.path.insert(0, ...) found in the existing CLI scripts.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ── FastAPI & middleware imports ─────────────────────────────────────────────
from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from sqlalchemy import text

from backend.app.core.config import settings
from backend.app.db.session import engine
from backend.app.schemas.error import ErrorDetail, ErrorResponse
from backend.app.api.routes import auth as auth_router
from backend.app.api.routes import health as health_router
from backend.app.api.routes import documents as documents_router
from backend.app.api.routes import summarize as summarize_router

# ── OpenAPI Tags Metadata ───────────────────────────────────────────────────
TAGS_METADATA = [
    {
        "name": "Health",
        "description": "Liveness probe and system health verification.",
    },
    {
        "name": "Authentication",
        "description": "User registration, login, and JWT access tokens.",
    },
    {
        "name": "Documents",
        "description": "Authenticated document upload, history, and ownership-scoped access.",
    },
    {
        "name": "Summarization",
        "description": "AI-powered legal and statutory tax document summarization.",
    },
]

# ── Application Lifespan ────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifecycle hook — lightweight validation only.

    Heavy pipeline components (FAISS index, embedding model, LLM client) are
    intentionally NOT loaded here. They initialise on first request so the
    server starts instantly and the health endpoint is always fast.
    """
    print(f"[startup] {settings.APP_NAME} v{settings.APP_VERSION}")
    print(f"[startup] environment : {settings.APP_ENV}")
    print(f"[startup] project root: {_PROJECT_ROOT}")
    print(f"[startup] CORS origins: {settings.CORS_ORIGINS}")
    try:
        with engine.connect() as conn:
            db_name = conn.execute(text("SELECT current_database()")).scalar()
        print(f"[startup] postgresql  : {db_name}")
        if db_name != "legal_summarizer":
            print(
                "[startup] WARNING: connected database is not 'legal_summarizer'. "
                "Check DATABASE_URL in the project-root .env."
            )
    except Exception as exc:  # noqa: BLE001
        print(f"[startup] WARNING: PostgreSQL connection failed: {type(exc).__name__}: {exc}")
    print("[startup] API docs available at /docs")
    yield
    print(f"[shutdown] {settings.APP_NAME} shutting down.")


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
    openapi_tags=TAGS_METADATA,
    lifespan=lifespan,
)

# ── CORS ─────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Standardized Exception Handlers ──────────────────────────────────────────
def _default_code_for_status(status_code: int) -> str:
    mapping = {
        400: "BAD_REQUEST",
        401: "UNAUTHORIZED",
        403: "FORBIDDEN",
        404: "NOT_FOUND",
        409: "CONFLICT",
        405: "METHOD_NOT_ALLOWED",
        413: "FILE_TOO_LARGE",
        415: "UNSUPPORTED_MEDIA_TYPE",
        422: "UNPROCESSABLE_ENTITY",
        500: "INTERNAL_SERVER_ERROR",
        502: "PROVIDER_ERROR",
        503: "SERVICE_UNAVAILABLE",
    }
    return mapping.get(status_code, f"HTTP_{status_code}")


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    """Standardize all HTTP exceptions into the unified ErrorResponse schema."""
    code = _default_code_for_status(exc.status_code)
    message = "An error occurred"
    details = None

    if isinstance(exc.detail, dict):
        if "code" in exc.detail:
            code = exc.detail["code"]
            message = exc.detail.get("message", "Request failed")
            details = exc.detail.get("details")
        elif "error" in exc.detail:
            # Backward-compatible adaptation of legacy error dictionary
            message = exc.detail.get("error") or "Request failed"
            details = exc.detail.get("detail")
        else:
            message = str(exc.detail)
    elif isinstance(exc.detail, str):
        message = exc.detail
    else:
        message = str(exc.detail)

    payload = ErrorResponse(
        success=False,
        error=ErrorDetail(code=code, message=message, details=details),
    )
    return JSONResponse(
        status_code=exc.status_code,
        content=payload.model_dump(),
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Standardize FastAPI / Pydantic validation errors (422) into ErrorResponse."""
    validation_issues = []
    for err in exc.errors():
        loc_parts = [str(part) for part in err.get("loc", []) if part != "body"]
        field_path = ".".join(loc_parts) if loc_parts else "body"
        msg = err.get("msg", "Invalid value")
        validation_issues.append({"field": field_path, "issue": msg})

    payload = ErrorResponse(
        success=False,
        error=ErrorDetail(
            code="VALIDATION_ERROR",
            message="Request validation failed",
            details=validation_issues,
        ),
    )
    return JSONResponse(
        status_code=422,
        content=payload.model_dump(),
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Catch-all exception handler. Logs internal details server-side while
    never exposing stack traces, internal paths, or API keys to the client.
    """
    print(f"[unhandled error] {type(exc).__name__}: {exc}")
    payload = ErrorResponse(
        success=False,
        error=ErrorDetail(
            code="INTERNAL_SERVER_ERROR",
            message="An unexpected server error occurred while processing the request.",
            details=None,
        ),
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=payload.model_dump(),
    )


# ── Routers ───────────────────────────────────────────────────────────────────
# All routes are versioned under /api/v1 for clean forward-compatibility.
API_PREFIX = "/api/v1"

app.include_router(health_router.router, prefix=API_PREFIX)
app.include_router(auth_router.router, prefix=API_PREFIX)
app.include_router(documents_router.router, prefix=API_PREFIX)
app.include_router(summarize_router.router, prefix=API_PREFIX)


# ── Convenience redirect ──────────────────────────────────────────────────────
@app.get("/", include_in_schema=False)
async def root_redirect() -> RedirectResponse:
    """Redirect bare root to the interactive API docs."""
    return RedirectResponse(url="/docs")


