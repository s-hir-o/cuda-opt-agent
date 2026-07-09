"""SQLAlchemy engine and session management.

SQLite-default, Postgres-ready. Uses WAL mode and optimized pragmas for SQLite.
Thread-safe: each call to get_session() produces an independent session.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Generator

from sqlalchemy import create_engine, event, Engine
from sqlalchemy.orm import Session, sessionmaker, DeclarativeBase
from sqlalchemy.pool import StaticPool

from cuda_opt_agent.core.config import DatabaseConfig


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""
    pass


_engine: Engine | None = None
_SessionFactory: sessionmaker[Session] | None = None


def _apply_sqlite_pragmas(dbapi_conn, connection_record) -> None:  # noqa: ANN001
    """Set performance pragmas on every SQLite connection."""
    cursor = dbapi_conn.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA synchronous=NORMAL")
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.execute("PRAGMA cache_size=-64000")   # 64 MB
    cursor.execute("PRAGMA temp_store=MEMORY")
    cursor.close()


def init_db(config: DatabaseConfig) -> Engine:
    """Initialize database engine, register ORM models, create tables.

    Args:
        config: Database configuration (URL, echo flag).

    Returns:
        The created SQLAlchemy Engine.
    """
    global _engine, _SessionFactory

    engine_kwargs = {
        "echo": config.echo,
        "pool_pre_ping": True,
    }
    if config.url.startswith("sqlite"):
        engine_kwargs["connect_args"] = {"check_same_thread": False}
        if config.url in {"sqlite:///:memory:", "sqlite://"}:
            engine_kwargs["poolclass"] = StaticPool

    # FIX: ensure the parent directory exists for SQLite file-based databases
    if config.url.startswith("sqlite"):
        db_path = config.url.replace("sqlite:///", "", 1)
        if db_path and db_path not in (":memory:", ""):
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    _engine = create_engine(config.url, **engine_kwargs)

    if config.url.startswith("sqlite"):
        event.listen(_engine, "connect", _apply_sqlite_pragmas)

    _SessionFactory = sessionmaker(
        bind=_engine,
        expire_on_commit=False,
    )

    # Import models so Base.metadata knows about them
    from cuda_opt_agent.memory import models  # noqa: F401

    Base.metadata.create_all(_engine)
    return _engine


def get_engine() -> Engine:
    """Return the active engine. Raises if not initialized."""
    if _engine is None:
        raise RuntimeError("Database not initialized. Call init_db() first.")
    return _engine


@contextmanager
def get_session() -> Generator[Session, None, None]:
    """Provide a transactional session scope.

    Commits on clean exit, rolls back on exception.
    Session is closed on exit regardless.
    """
    if _SessionFactory is None:
        raise RuntimeError("Database not initialized. Call init_db() first.")

    session = _SessionFactory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def dispose_engine() -> None:
    """Dispose the engine and clear global state. For clean shutdown / testing."""
    global _engine, _SessionFactory
    if _engine is not None:
        _engine.dispose()
        _engine = None
        _SessionFactory = None