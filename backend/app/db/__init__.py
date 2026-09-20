"""
Database package initialization.

Exports SQLAlchemy Base, engine, session maker, get_db dependency,
and database models.
"""

from __future__ import annotations

from backend.app.db.models import Document, Summary
from backend.app.db.session import Base, SessionLocal, engine, get_db

__all__ = [
    "Base",
    "engine",
    "SessionLocal",
    "get_db",
    "Document",
    "Summary",
]
