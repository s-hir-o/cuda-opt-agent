"""Dependency injection for FastAPI endpoints.

Provides database sessions, repository instances, and service objects
as FastAPI dependencies. Sessions are scoped per-request.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request
from sqlalchemy.orm import Session

from cuda_opt_agent.core.config import AppSettings
from cuda_opt_agent.memory.database import get_session
from cuda_opt_agent.memory.repositories.kernel_repo import KernelRepo
from cuda_opt_agent.memory.repositories.metrics_repo import MetricsRepo
from cuda_opt_agent.memory.repositories.result_repo import ResultRepo
from cuda_opt_agent.memory.repositories.run_repo import RunRepo


def get_settings(request: Request) -> AppSettings:
    return request.app.state.settings


def get_db_session():
    """Yield a transactional database session. Commits on success, rolls back on error."""
    with get_session() as session:
        yield session


def get_kernel_repo(session: Session = Depends(get_db_session)) -> KernelRepo:
    return KernelRepo(session)


def get_run_repo(session: Session = Depends(get_db_session)) -> RunRepo:
    return RunRepo(session)


def get_metrics_repo(session: Session = Depends(get_db_session)) -> MetricsRepo:
    return MetricsRepo(session)


def get_result_repo(session: Session = Depends(get_db_session)) -> ResultRepo:
    return ResultRepo(session)


SettingsDep = Annotated[AppSettings, Depends(get_settings)]
SessionDep = Annotated[Session, Depends(get_db_session)]
KernelRepoDep = Annotated[KernelRepo, Depends(get_kernel_repo)]
RunRepoDep = Annotated[RunRepo, Depends(get_run_repo)]
MetricsRepoDep = Annotated[MetricsRepo, Depends(get_metrics_repo)]
ResultRepoDep = Annotated[ResultRepo, Depends(get_result_repo)]
