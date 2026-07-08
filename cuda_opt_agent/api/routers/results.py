"""Results query endpoints — metrics, benchmarks, compilations."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from cuda_opt_agent.api.dependencies import (
    KernelRepoDep,
    MetricsRepoDep,
    ResultRepoDep,
    RunRepoDep,
)
from cuda_opt_agent.schemas.benchmark import BenchmarkResult

router = APIRouter()


# ── Response models ────────────────────────────────────────────

class MetricsResponse(BaseModel):
    kernel_name: str
    gpu_time_us: float | None
    sm_throughput: float | None
    dram_throughput: float | None
    achieved_occupancy: float | None
    registers_per_thread: int | None


class BenchmarkSummaryResponse(BaseModel):
    variant_id: str
    mean_ms: float
    median_ms: float
    std_ms: float
    speedup_vs_baseline: float | None


class CompilationResponse(BaseModel):
    success: bool
    arch: str
    compile_time_s: float
    error_count: int
    warning_count: int


class RunSummaryResponse(BaseModel):
    run_id: str
    kernel_name: str
    status: str
    iterations: int
    best_speedup: float
    best_variant_id: str | None


# ── Endpoints ──────────────────────────────────────────────────

@router.get("/metrics/{variant_id}")
async def get_metrics(variant_id: str, repo: MetricsRepoDep):
    """Get profiling metrics for a variant."""
    metrics = repo.find_by_variant(variant_id)
    if metrics is None:
        raise HTTPException(status_code=404, detail=f"No metrics for variant '{variant_id}'")

    return MetricsResponse(
        kernel_name=metrics.kernel_name,
        gpu_time_us=metrics.gpu_time_us,
        sm_throughput=metrics.sm_throughput,
        dram_throughput=metrics.dram_throughput,
        achieved_occupancy=metrics.achieved_occupancy,
        registers_per_thread=metrics.registers_per_thread,
    )


@router.get("/benchmarks/{variant_id}")
async def get_benchmarks(variant_id: str, repo: ResultRepoDep):
    """Get benchmark results for a variant."""
    benchmarks = repo.find_benchmarks_by_variant(variant_id)
    if not benchmarks:
        raise HTTPException(status_code=404, detail=f"No benchmarks for variant '{variant_id}'")

    return [
        BenchmarkSummaryResponse(
            variant_id=b.variant_id,
            mean_ms=b.timing.mean_ms,
            median_ms=b.timing.median_ms,
            std_ms=b.timing.std_ms,
            speedup_vs_baseline=None,
        )
        for b in benchmarks
    ]


@router.get("/compilation/{variant_id}")
async def get_compilation(variant_id: str, run_repo: RunRepoDep):
    """Get compilation result for a variant.

    Searches through run iterations to find the compilation for this variant.
    """
    # Note: In a full implementation, there would be a direct lookup by variant_id.
    # For now, return 404 with guidance.
    raise HTTPException(
        status_code=501,
        detail="Direct compilation lookup not yet implemented. Use GET /runs/{run_id} for iteration details.",
    )


@router.get("/history")
async def get_history(
    kernel_repo: KernelRepoDep,
    run_repo: RunRepoDep,
    kernel_name: str | None = None,
    offset: int = 0,
    limit: int = 20,
):
    """Get optimization history, optionally filtered by kernel name."""
    runs = run_repo.list_runs(offset=offset, limit=limit)

    if kernel_name:
        kernels = kernel_repo.find_sources_by_name(kernel_name)
        kernel_ids = set()
        # Collect kernel IDs by matching name
        for k in kernels:
            # Source objects don't have DB IDs directly — use run filtering
            pass
        # Filter runs by kernel name
        runs = [r for r in runs if r.kernel.name == kernel_name]

    return [
        RunSummaryResponse(
            run_id=r.run_id,
            kernel_name=r.kernel.name,
            status=r.status,
            iterations=r.iteration_count,
            best_speedup=r.best_speedup,
            best_variant_id=r.best_variant_id,
        )
        for r in runs
    ]
