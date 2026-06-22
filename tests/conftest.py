"""Shared test fixtures."""

from __future__ import annotations

import textwrap

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from cuda_opt_agent.core.config import AppSettings
from cuda_opt_agent.memory.database import Base
from cuda_opt_agent.memory import models  # noqa: F401 — register ORM models
from cuda_opt_agent.schemas.kernel import KernelLanguage, KernelSource


@pytest.fixture
def default_settings() -> AppSettings:
    return AppSettings()


@pytest.fixture
def sample_kernel_source() -> KernelSource:
    return KernelSource(
        name="naive_matmul.cu",
        language=KernelLanguage.CUDA,
        source_code=textwrap.dedent("""\
            __global__ void matmul(float *C, const float *A, const float *B, int N) {
                int row = blockIdx.y * blockDim.y + threadIdx.y;
                int col = blockIdx.x * blockDim.x + threadIdx.x;
                if (row < N && col < N) {
                    float sum = 0.0f;
                    for (int k = 0; k < N; k++) {
                        sum += A[row * N + k] * B[k * N + col];
                    }
                    C[row * N + col] = sum;
                }
            }
        """),
        entry_point="matmul",
    )


@pytest.fixture
def db_engine():
    """In-memory SQLite engine with all tables created."""
    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(engine)
    yield engine
    engine.dispose()


@pytest.fixture
def db_session(db_engine) -> Session:
    """Transactional session that rolls back after each test."""
    SessionLocal = sessionmaker(bind=db_engine, expire_on_commit=False)
    session = SessionLocal()
    yield session
    session.rollback()
    session.close()


@pytest.fixture
def db_session_factory(db_engine):
    """Session factory for tests that need multiple independent sessions."""
    return sessionmaker(bind=db_engine, expire_on_commit=False)