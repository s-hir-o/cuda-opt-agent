"""FastAPI application factory.

Creates and configures the FastAPI app with routers, middleware,
and lifespan management (database init/shutdown).
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from cuda_opt_agent.api.middleware.logging import RequestLoggingMiddleware
from cuda_opt_agent.api.middleware.tracing import TraceIDMiddleware
from cuda_opt_agent.api.routers import kernels, runs, results
from cuda_opt_agent.core.config import AppSettings, load_settings
from cuda_opt_agent.memory.database import dispose_engine, init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: init DB on startup, dispose on shutdown."""
    settings: AppSettings = app.state.settings
    init_db(settings.database)
    yield
    dispose_engine()


def create_app(settings: AppSettings | None = None) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        settings: App configuration. If None, loads from default config.yaml.

    Returns:
        Configured FastAPI instance.
    """
    if settings is None:
        settings = load_settings("config.yaml")

    app = FastAPI(
        title="CUDA Optimization Agent",
        description="REST API for CUDA kernel auto-optimization",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.state.settings = settings

    # Middleware (order matters — outermost first)
    app.add_middleware(TraceIDMiddleware)
    app.add_middleware(RequestLoggingMiddleware)

    # Routers
    app.include_router(kernels.router, prefix="/api/v1/kernels", tags=["kernels"])
    app.include_router(runs.router, prefix="/api/v1/runs", tags=["runs"])
    app.include_router(results.router, prefix="/api/v1/results", tags=["results"])

    @app.get("/health")
    async def health():
        return {"status": "ok", "version": "0.1.0"}

    return app
