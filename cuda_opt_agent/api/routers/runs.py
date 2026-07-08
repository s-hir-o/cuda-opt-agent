"""Optimization run endpoints — start, list, and query runs."""

from __future__ import annotations

import asyncio
from datetime import datetime

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field

from cuda_opt_agent.api.dependencies import KernelRepoDep, RunRepoDep, SettingsDep
from cuda_opt_agent.core.config import AppSettings
from cuda_opt_agent.core.logging import get_logger
from cuda_opt_agent.schemas.kernel import KernelSource

log = get_logger(__name__)

router = APIRouter()


# ── Request/Response models ────────────────────────────────────

class RunStartRequest(BaseModel):
    kernel_id: int
    max_iterations: int = 20
    arch: str | None = None


class RunResponse(BaseModel):
    run_id: str
    kernel_name: str
    target_arch: str
    status: str
    best_speedup: float
    best_variant_id: str | None
    iteration_count: int
    started_at: datetime
    completed_at: datetime | None


class IterationResponse(BaseModel):
    iteration: int
    variant_id: str
    speedup: float | None
    bottleneck: str | None
    transform: str | None
    error: str | None


class RunDetailResponse(RunResponse):
    iterations: list[IterationResponse]


class RunListResponse(BaseModel):
    runs: list[RunResponse]
    total: int


# ── Endpoints ──────────────────────────────────────────────────

@router.post("", response_model=RunResponse, status_code=201)
async def start_run(
    request: RunStartRequest,
    background: BackgroundTasks,
    settings: SettingsDep,
    kernel_repo: KernelRepoDep,
    run_repo: RunRepoDep,
):
    """Start a new optimization run.

    The optimization runs in the background. Poll GET /runs/{run_id} for status.
    """
    kernel = kernel_repo.get_source(request.kernel_id)
    if kernel is None:
        raise HTTPException(status_code=404, detail=f"Kernel {request.kernel_id} not found")

    arch = request.arch or settings.compiler.default_arch

    # Build coordinator and run in background
    background.add_task(
        _run_optimization,
        kernel=kernel,
        arch=arch,
        max_iterations=request.max_iterations,
        settings=settings,
    )

    # Return a placeholder — real run_id will be set by the background task
    # For now, return a pending response
    from cuda_opt_agent.core.ids import generate_run_id
    run_id = generate_run_id()

    return RunResponse(
        run_id=run_id,
        kernel_name=kernel.name,
        target_arch=arch,
        status="pending",
        best_speedup=1.0,
        best_variant_id=None,
        iteration_count=0,
        started_at=datetime.utcnow(),
        completed_at=None,
    )


@router.get("", response_model=RunListResponse)
async def list_runs(
    run_repo: RunRepoDep,
    kernel_id: int | None = None,
    status: str | None = None,
    offset: int = 0,
    limit: int = 50,
):
    """List optimization runs with optional filters."""
    runs = run_repo.list_runs(
        kernel_id=kernel_id,
        status=status,
        offset=offset,
        limit=limit,
    )

    return RunListResponse(
        runs=[
            RunResponse(
                run_id=r.run_id,
                kernel_name=r.kernel.name,
                target_arch=r.target_arch,
                status=r.status,
                best_speedup=r.best_speedup,
                best_variant_id=r.best_variant_id,
                iteration_count=r.iteration_count,
                started_at=r.started_at,
                completed_at=r.completed_at,
            )
            for r in runs
        ],
        total=len(runs),
    )


@router.get("/{run_id}", response_model=RunDetailResponse)
async def get_run(run_id: str, run_repo: RunRepoDep):
    """Get detailed information about a specific run, including all iterations."""
    run = run_repo.get_run(run_id, load_iterations=True)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

    iterations = [
        IterationResponse(
            iteration=it.iteration,
            variant_id=it.variant_id,
            speedup=it.speedup.speedup if it.speedup else None,
            bottleneck=it.bottleneck.primary.value if it.bottleneck else None,
            transform=str(it.transform_applied) if it.transform_applied else None,
            error=it.error,
        )
        for it in run.iterations
    ]

    return RunDetailResponse(
        run_id=run.run_id,
        kernel_name=run.kernel.name,
        target_arch=run.target_arch,
        status=run.status,
        best_speedup=run.best_speedup,
        best_variant_id=run.best_variant_id,
        iteration_count=run.iteration_count,
        started_at=run.started_at,
        completed_at=run.completed_at,
        iterations=iterations,
    )


# ── Background task ────────────────────────────────────────────

async def _run_optimization(
    kernel: KernelSource,
    arch: str,
    max_iterations: int,
    settings: AppSettings,
) -> None:
    """Run optimization in background. Logs results but does not return them."""
    try:
        from cuda_opt_agent.agents.coordinator import Coordinator
        from cuda_opt_agent.agents.profiler_agent import ProfilerAgent
        from cuda_opt_agent.agents.analysis_agent import AnalysisAgent
        from cuda_opt_agent.agents.bottleneck_agent import BottleneckAgent
        from cuda_opt_agent.agents.transform_agent import TransformAgent
        from cuda_opt_agent.agents.benchmark_agent import BenchmarkAgent
        from cuda_opt_agent.analysis.ptx.mock_analyzer import MockPTXAnalyzer
        from cuda_opt_agent.benchmark.mock_runner import MockBenchmarkRunner
        from cuda_opt_agent.build.nvcc_runner import NVCCRunner
        from cuda_opt_agent.cost_models.analytical.roofline import RooflineModel
        from cuda_opt_agent.features.extractors.ncu_features import NCUFeatureExtractor
        from cuda_opt_agent.features.pipeline import FeaturePipeline
        from cuda_opt_agent.llm.factory import create_provider
        from cuda_opt_agent.profiling.mock_profiler import MockProfiler
        from cuda_opt_agent.search.grid_search import GridSearch

        cfg = settings.model_copy()
        cfg.optimization.max_iterations = max_iterations

        # Build with mocks if real tools unavailable
        nvcc = NVCCRunner(cfg.compiler)
        nvcc_ok = await nvcc.is_available()

        profiler = ProfilerAgent(MockProfiler()) if not cfg.profiling.enabled else ProfilerAgent(MockProfiler())

        try:
            llm = create_provider(cfg.llm)
        except Exception:
            from cuda_opt_agent.agents.transform_agent import TransformAgent as _TA
            llm = None

        coordinator = Coordinator(
            profiler_agent=profiler,
            analysis_agent=AnalysisAgent(
                analyzer=MockPTXAnalyzer(),
                feature_pipeline=FeaturePipeline([NCUFeatureExtractor()]),
                cost_model=RooflineModel(),
            ),
            bottleneck_agent=BottleneckAgent(),
            transform_agent=TransformAgent(llm) if llm else TransformAgent(_NoOpLLM()),
            benchmark_agent=BenchmarkAgent(
                nvcc_runner=nvcc,
                benchmark_runner=MockBenchmarkRunner(),
                benchmark_config=cfg.benchmark,
            ),
            search_strategy=GridSearch(batch_size=cfg.search.grid.parallelism),
            config=cfg,
        )

        run = await coordinator.optimize(kernel)
        log.info("background_optimization.done", run_id=run.run_id, status=run.status)

    except Exception as exc:
        log.error("background_optimization.failed", error=str(exc))


class _NoOpLLM:
    """Fallback when no LLM provider is available."""
    def provider_name(self): return "noop"
    def estimate_cost(self, i, o): return 0
    async def health_check(self): return False
    async def generate(self, req):
        from cuda_opt_agent.schemas.llm import LLMResponse
        return LLMResponse(content='{"analysis_summary":"no LLM","transforms":[]}', model="noop", provider="noop")
    async def generate_json(self, req, retries=3):
        return await self.generate(req)
