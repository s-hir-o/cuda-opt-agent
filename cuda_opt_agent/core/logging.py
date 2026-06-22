"""Structured logging setup via structlog with run_id injection."""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog

from cuda_opt_agent.core.config import LoggingConfig


_configured = False


def setup_logging(config: LoggingConfig, run_id: str | None = None) -> None:
    """Configure structlog + stdlib logging once.

    Args:
        config: Logging configuration from AppSettings.
        run_id: Optional run_id to inject into all log entries.
    """
    global _configured
    if _configured:
        return

    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    if config.format == "json":
        renderer: structlog.types.Processor = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
        foreign_pre_chain=shared_processors,
    )

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(config.level.upper())

    if config.file:
        file_handler = logging.FileHandler(config.file)
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    if run_id:
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(run_id=run_id)

    _configured = True


def get_logger(name: str | None = None, **initial_context: Any) -> structlog.stdlib.BoundLogger:
    """Get a bound logger instance.

    Args:
        name: Logger name (typically __name__).
        **initial_context: Additional key-value pairs bound to this logger.

    Returns:
        Configured structlog BoundLogger.
    """
    return structlog.get_logger(name, **initial_context)  # type: ignore[return-value]
