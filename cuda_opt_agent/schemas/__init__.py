"""All Pydantic data models shared across the platform."""

from cuda_opt_agent.schemas.analysis import (
    DependencyEdge,
    DependencyGraph,
    KernelAnalysis,
    MemAccessPattern,
    PTXInstruction,
)
from cuda_opt_agent.schemas.bottleneck import BottleneckClass, BottleneckResult
from cuda_opt_agent.schemas.benchmark import BenchmarkResult, SpeedupResult, TimingStats
from cuda_opt_agent.schemas.compilation import CompilationDiagnostic, CompilationResult
from cuda_opt_agent.schemas.correctness import CorrectnessResult
from cuda_opt_agent.schemas.cost_model import BoundType, CostEstimate, ModelPrediction
from cuda_opt_agent.schemas.features import KernelFeatureVector
from cuda_opt_agent.schemas.kernel import KernelLanguage, KernelSource, KernelVariant
from cuda_opt_agent.schemas.llm import LLMRequest, LLMResponse
from cuda_opt_agent.schemas.metrics import ProfileMetrics
from cuda_opt_agent.schemas.optimization import OptimizationIteration, OptimizationRun
from cuda_opt_agent.schemas.transform import TransformBackend, TransformResult, TransformSpec

__all__ = [
    # kernel
    "KernelLanguage",
    "KernelSource",
    "KernelVariant",
    # metrics
    "ProfileMetrics",
    # bottleneck
    "BottleneckClass",
    "BottleneckResult",
    # transform
    "TransformBackend",
    "TransformSpec",
    "TransformResult",
    # compilation
    "CompilationResult",
    "CompilationDiagnostic",
    # benchmark
    "BenchmarkResult",
    "SpeedupResult",
    "TimingStats",
    # correctness
    "CorrectnessResult",
    # optimization
    "OptimizationRun",
    "OptimizationIteration",
    # llm
    "LLMRequest",
    "LLMResponse",
    # analysis
    "PTXInstruction",
    "DependencyEdge",
    "DependencyGraph",
    "MemAccessPattern",
    "KernelAnalysis",
    # features
    "KernelFeatureVector",
    # cost model
    "CostEstimate",
    "ModelPrediction",
    "BoundType",
]
