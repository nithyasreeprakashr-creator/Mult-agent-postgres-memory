from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Generator

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, declarative_base, sessionmaker
from sqlalchemy.pool import QueuePool

load_dotenv()

# Connection settings (all overridable via .env)

POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "postgres")
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")
POSTGRES_DB = os.getenv("POSTGRES_DB", "software_factory_memory")

# SQLAlchemy URL (uses the psycopg3 driver: postgresql+psycopg://)
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    f"postgresql+psycopg://{POSTGRES_USER}:{POSTGRES_PASSWORD}"
    f"@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}",
)

# Plain DSN (no SQLAlchemy prefix) — required by langgraph's PostgresSaver,
# which talks to psycopg directly instead of through SQLAlchemy.
CHECKPOINT_DATABASE_URL = os.getenv(
    "CHECKPOINT_DATABASE_URL",
    f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}"
    f"@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}",
)

# Engine / Session / Base

engine = create_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,   # avoids stale-connection errors
    pool_recycle=1800,
    future=True,
)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
    future=True,
)

Base = declarative_base()


# Helpers

def init_db() -> None:
    """
    Create all ORM tables if they do not already exist.
    Safe to call on every startup (idempotent).
    """
    import models  # noqa: F401  (registers ProjectMemory on Base.metadata)

    Base.metadata.create_all(bind=engine)


@contextmanager
def get_db_session() -> Generator[Session, None, None]:
    """
    Context-managed session.

    Usage:
        with get_db_session() as session:
            session.add(obj)
    Automatically commits on success and rolls back on exception.
    """
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def check_connection() -> bool:
    """Lightweight health check used before running the workflow."""
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception as exc:
        print(f"   ⚠  PostgreSQL connection failed: {exc}")
        return False
