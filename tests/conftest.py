"""Shared test fixtures."""

import os
import tempfile

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import NullPool

from cuda_opt_agent.memory.database import Base
import cuda_opt_agent.memory.models  # noqa: F401


@pytest.fixture()
def db_engine():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    url = f"sqlite:///{path}"

    engine = create_engine(
        url,
        poolclass=NullPool,
        connect_args={"check_same_thread": False},
    )

    @event.listens_for(engine, "connect")
    def _on_connect(dbapi_conn, connection_record):
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.execute("PRAGMA busy_timeout=5000")
        cursor.close()

    Base.metadata.create_all(engine)

    yield engine

    engine.dispose()
    try:
        os.unlink(path)
    except OSError:
        pass


@pytest.fixture()
def db_session_factory(db_engine):
    return sessionmaker(
        bind=db_engine,
        class_=Session,
        # autoflush=True (default) so queries within the same session
        # see pending inserts.  Session isolation still holds because
        # each session gets its own connection via NullPool.
        expire_on_commit=False,
    )


@pytest.fixture()
def db_session(db_session_factory):
    session = db_session_factory()
    try:
        yield session
    finally:
        session.rollback()
        session.close()