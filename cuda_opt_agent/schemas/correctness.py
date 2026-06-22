"""Correctness validation result."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class CorrectnessResult(BaseModel):
    """Result of comparing variant output against baseline."""

    passed: bool
    variant_id: str
    baseline_variant_id: str
    max_absolute_error: float = 0.0
    max_relative_error: float = 0.0
    tolerance_absolute: float = 1e-5
    tolerance_relative: float = 1e-4
    failed_elements: int = 0
    total_elements: int = 0
    message: str = ""
    checked_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def failure_rate(self) -> float:
        if self.total_elements == 0:
            return 0.0
        return self.failed_elements / self.total_elements
