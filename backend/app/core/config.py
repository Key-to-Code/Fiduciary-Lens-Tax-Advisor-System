"""
Backend configuration using pydantic-settings.

Reads from the .env file at the project root (two levels above this file).
This module handles only the web-server / CORS layer; the RAG pipeline's
own settings (embed model, LLM provider, index path, etc.) continue to be
managed by shared/config.py exactly as before.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict

# Project root: backend/app/core/ -> backend/app/ -> backend/ -> project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


class Settings(BaseSettings):
    """Application settings loaded from environment / .env file."""

    # ── Application identity ─────────────────────────────────────────────────
    APP_NAME: str = "Fiduciary Lens Tax Advisor API"
    APP_VERSION: str = "0.1.0"
    APP_ENV: str = "development"  # development | staging | production

    # ── Server ───────────────────────────────────────────────────────────────
    BACKEND_HOST: str = "0.0.0.0"
    BACKEND_PORT: int = 8000

    # ── CORS ─────────────────────────────────────────────────────────────────
    # Comma-separated list of allowed origins.
    # Defaults cover the design-mockup HTML served locally and common
    # frontend dev-server ports (Vite 5173, CRA 3000, generic 8080).
    CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:8080",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:8080",
    ]

    model_config = SettingsConfigDict(
        # Load from .env at the project root; ignore missing file gracefully.
        env_file=str(_PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        # Allow comma-separated strings for list fields (e.g. CORS_ORIGINS=a,b,c).
        env_ignore_empty=True,
        extra="ignore",
    )


# Module-level singleton — imported as `from backend.app.core.config import settings`
settings = Settings()
