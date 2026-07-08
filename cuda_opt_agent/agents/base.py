"""Agent context — mutable state carrier for the optimization pipeline.

Each agent reads from and writes to the context.
The coordinator owns the context and passes it between agents.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from cuda_opt_agent.schemas.analysis import KernelAnalysis
from cuda_opt_agent.schemas.benchmark import BenchmarkResult, SpeedupResult
from cuda_opt_agent.schemas.bottleneck import BottleneckResult
from cuda_opt_agent.schemas.compilation import CompilationResult
from cuda_opt_agent.schemas.cost_model import CostEstimate
from cuda_opt_agent.schemas.correctness import CorrectnessResult
from cuda_opt_agent.schemas.features import KernelFeatureVector
from cuda_opt_agent.schemas.kernel import KernelSource, KernelVariant
from cuda_opt_agent.schemas.metrics import ProfileMetrics


@dataclass
class AgentContext:
    """Mutable state passed between pipeline agents.

    Populated progressively as the pipeline executes.
    Any field may be None if that stage has not yet run or failed.
    """

    run_id: str
    kernel: KernelSource | KernelVariant
    arch: str

    # Pipeline outputs — populated in order
    profile_metrics: ProfileMetrics | None = None
    ptx_source: str | None = None
    analysis: KernelAnalysis | None = None
    features: KernelFeatureVector | None = None
    cost_estimate: CostEstimate | None = None
    bottleneck: BottleneckResult | None = None
    compilation: CompilationResult | None = None
    correctness: CorrectnessResult | None = None
    benchmark: BenchmarkResult | None = None
    speedup: SpeedupResult | None = None

    # Error tracking
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def add_error(self, msg: str) -> None:
        self.errors.append(msg)

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)

    @property
    def has_errors(self) -> bool:
        return len(self.errors) > 0
