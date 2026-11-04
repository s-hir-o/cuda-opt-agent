"""Optimization run and iteration models."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from cuda_opt_agent.schemas.benchmark import BenchmarkResult, SpeedupResult
from cuda_opt_agent.schemas.bottleneck import BottleneckResult
from cuda_opt_agent.schemas.compilation import CompilationResult
from cuda_opt_agent.schemas.correctness import CorrectnessResult
from cuda_opt_agent.schemas.kernel import KernelSource, KernelVariant
from cuda_opt_agent.schemas.metrics import ProfileMetrics
from cuda_opt_agent.schemas.transform import TransformResult, TransformSpec


class OptimizationIteration(BaseModel):
    """A single iteration of the optimization loop."""

    iteration: int
    variant_id: str
    variant: KernelVariant

    # Pipeline results (each optional — pipeline may not have completed all stages)
    profile_metrics: ProfileMetrics | None = None
    bottleneck: BottleneckResult | None = None
    transform_applied: TransformSpec | None = None
    transform_result: TransformResult | None = None
    compilation: CompilationResult | None = None
    correctness: CorrectnessResult | None = None
    benchmark: BenchmarkResult | None = None
    speedup: SpeedupResult | None = None

    # NEW: optional enrichment (v2)
    analysis: Any = None         # KernelAnalysis (avoid circular import)
    features: Any = None         # KernelFeatureVector
    cost_estimate: Any = None    # CostEstimate

    started_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: datetime | None = None
    error: str | None = None

    @property
    def is_complete(self) -> bool:
        return self.completed_at is not None and self.error is None

    @property
    def achieved_speedup(self) -> float:
        return self.speedup.speedup if self.speedup else 1.0

    @property
    def validated_speedup(self) -> float:
        """Speedup eligible for best-variant selection.

        Raw speedups are useful for reporting, but a candidate should only be
        selected as best when the benchmark comparator marked it as both
        statistically significant and above the configured effect-size floor.
        """
        if self.speedup is None:
            return 1.0
        if not self.speedup.significant:
            return 1.0
        if not self.speedup.is_improvement:
            return 1.0
        return self.speedup.effective_speedup


class OptimizationRun(BaseModel):
    """Full optimization run tracking all iterations."""

    run_id: str
    kernel: KernelSource
    target_arch: str
    max_iterations: int = 20

    iterations: list[OptimizationIteration] = Field(default_factory=list)
    best_variant_id: str | None = None
    best_speedup: float = 1.0

    started_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: datetime | None = None
    status: str = "running"  # running | converged | failed | cancelled

    config_snapshot: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def iteration_count(self) -> int:
        return len(self.iterations)

    @property
    def is_converged(self) -> bool:
        return self.status == "converged"

    def add_iteration(self, iteration: OptimizationIteration) -> None:
        self.iterations.append(iteration)
        if iteration.validated_speedup > self.best_speedup:
            self.best_speedup = iteration.validated_speedup
            self.best_variant_id = iteration.variant_id
