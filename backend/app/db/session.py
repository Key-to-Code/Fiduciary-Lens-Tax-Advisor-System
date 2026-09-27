"""
SQLAlchemy database engine and session configuration.

Manages connection pool, session factory, base declarative model,
and the FastAPI database dependency.
"""

from __future__ import annotations

from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from backend.app.core.config import settings


class Base(DeclarativeBase):
    """Base declarative class for all SQLAlchemy models."""
    pass


# Create database engine with connection pooling and liveness verification
engine = create_engine(
    settings.DATABASE_URL,
    pool_pre_ping=True,
    echo=False,
    # Keep the optional startup database probe from blocking the API when
    # PostgreSQL is stopped or unreachable. RAG and health routes are DB-free.
    connect_args={"connect_timeout": 3},
)

# Session factory for generating discrete transactional database sessions
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)


def get_db() -> Generator[Session, None, None]:
    """
    FastAPI dependency that provides a transactional database session per request.
    Rolls back on unhandled errors and always closes the session at request end.
    """
    db = SessionLocal()
    try:
        yield db
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
