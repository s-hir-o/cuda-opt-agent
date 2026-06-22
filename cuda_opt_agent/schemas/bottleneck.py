"""Bottleneck classification."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class BottleneckClass(str, Enum):
    MEMORY_BOUND = "memory_bound"
    COMPUTE_BOUND = "compute_bound"
    LATENCY_BOUND = "latency_bound"
    DIVERGENCE_BOUND = "divergence_bound"
    OCCUPANCY_BOUND = "occupancy_bound"
    UNCLASSIFIED = "unclassified"


class BottleneckResult(BaseModel):
    """Output of the bottleneck classifier agent."""

    primary: BottleneckClass
    confidence: float = Field(ge=0.0, le=1.0, description="Classification confidence 0-1")
    secondary: BottleneckClass | None = None
    reasoning: str = ""
    evidence: dict[str, Any] = Field(default_factory=dict)
    recommended_transforms: list[str] = Field(
        default_factory=list,
        description="Transform names recommended for this bottleneck",
    )
