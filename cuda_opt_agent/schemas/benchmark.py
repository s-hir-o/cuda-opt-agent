"""Benchmark timing and speedup results."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class TimingStats(BaseModel):
    """Statistical summary of benchmark timing runs."""

    mean_ms: float
    median_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    p95_ms: float | None = None
    p99_ms: float | None = None
    n_samples: int = 0
    raw_times_ms: list[float] = Field(default_factory=list)

    @property
    def cv(self) -> float:
        """Coefficient of variation (std/mean). Lower = more stable."""
        return self.std_ms / self.mean_ms if self.mean_ms > 0 else float("inf")


class BenchmarkResult(BaseModel):
    """Benchmark outcome for a single kernel variant."""

    variant_id: str
    timing: TimingStats
    arch: str = ""
    block_size: tuple[int, ...] | None = None
    grid_size: tuple[int, ...] | None = None
    correct: bool = True
    benchmarked_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)


class SpeedupResult(BaseModel):
    """Comparison of variant vs baseline."""

    baseline_variant_id: str
    candidate_variant_id: str
    baseline_time_ms: float
    candidate_time_ms: float
    speedup: float = Field(description="baseline_time / candidate_time (>1 = faster)")
    significant: bool = Field(
        description="True if speedup exceeds statistical noise"
    )
    confidence_interval: tuple[float, float] | None = None

    @property
    def is_improvement(self) -> bool:
        return self.speedup > 1.0
