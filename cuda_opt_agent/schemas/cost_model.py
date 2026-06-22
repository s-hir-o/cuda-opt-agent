"""Cost estimation and performance prediction models."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class BoundType(str, Enum):
    """What the kernel is bounded by (roofline model output)."""
    MEMORY = "memory"
    COMPUTE = "compute"
    LATENCY = "latency"
    MIXED = "mixed"
    UNKNOWN = "unknown"


class CostEstimate(BaseModel):
    """Predicted performance characteristics of a kernel."""

    model_name: str
    bound_type: BoundType = BoundType.UNKNOWN
    predicted_time_ms: float | None = None
    predicted_throughput_pct: float | None = None
    predicted_occupancy: float | None = None
    predicted_register_usage: int | None = None
    confidence: float = Field(
        default=0.5, ge=0.0, le=1.0,
        description="Model confidence in prediction (0-1)"
    )
    reasoning: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class ModelPrediction(BaseModel):
    """Predicted relative performance of variant vs baseline."""

    model_name: str
    predicted_speedup: float = Field(description="Predicted speedup (>1 = improvement)")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    reasoning: str = ""
    improvement_areas: list[str] = Field(default_factory=list)
    risk_areas: list[str] = Field(default_factory=list)

    @property
    def is_predicted_improvement(self) -> bool:
        return self.predicted_speedup > 1.0
