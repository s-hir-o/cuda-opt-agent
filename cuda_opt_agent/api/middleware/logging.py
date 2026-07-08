"""Request/response logging middleware.

Logs method, path, status code, and duration for every request.
Skips health check endpoint to reduce noise.
"""

from __future__ import annotations

import time

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

from cuda_opt_agent.core.logging import get_logger

log = get_logger("api.access")

_SKIP_PATHS = {"/health", "/docs", "/openapi.json", "/redoc"}


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log every API request with method, path, status, and duration."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        path = request.url.path

        if path in _SKIP_PATHS:
            return await call_next(request)

        t0 = time.monotonic()

        response = await call_next(request)

        duration_ms = (time.monotonic() - t0) * 1000

        log.info(
            "request",
            method=request.method,
            path=path,
            status=response.status_code,
            duration_ms=round(duration_ms, 1),
        )

        return response
