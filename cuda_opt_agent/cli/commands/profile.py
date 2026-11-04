"""`cuda-opt profile` command — profile a CUDA kernel."""

from __future__ import annotations

import asyncio
from pathlib import Path

import typer

from cuda_opt_agent.cli.output import print_error, print_info, print_success
from cuda_opt_agent.core.config import load_settings
from cuda_opt_agent.schemas.kernel import KernelLanguage, KernelSource


def profile_cmd(
    file: Path = typer.Argument(..., help="Path to CUDA source file (.cu)"),
    arch: str = typer.Option(None, "--arch", "-a", help="Target GPU architecture"),
    config: Path = typer.Option(None, "--config", "-c", help="Path to config.yaml"),
):
    """Profile a CUDA kernel and display metrics."""
    if not file.is_file():
        print_error(f"File not found: {file}")
        raise typer.Exit(code=1)

    source_code = file.read_text()
    settings = load_settings(config)
    if arch:
        settings.compiler.default_arch = arch

    kernel = KernelSource(
        name=file.name,
        language=KernelLanguage.CUDA,
        source_code=source_code,
        entry_point="main",
    )

    print_info(f"Profiling {file.name} (arch={settings.compiler.default_arch})")

    result = asyncio.run(_run_profile(kernel, settings))

    if result is None:
        print_error("Profiling failed")
        raise typer.Exit(code=1)

    print_success("\nProfiling complete:")
    _print_metrics(result)


async def _run_profile(kernel: KernelSource, settings):
    from cuda_opt_agent.agents.base import AgentContext
    from cuda_opt_agent.agents.profiler_agent import ProfilerAgent
    from cuda_opt_agent.build.nvcc_runner import NVCCRunner
    from cuda_opt_agent.core.ids import generate_run_id
    from cuda_opt_agent.profiling.ncu_runner import NCUProfiler, NCURunner

    nvcc = NVCCRunner(settings.compiler)
    ncu = NCURunner(settings.profiling)
    profiler = ProfilerAgent(NCUProfiler(ncu_runner=ncu, nvcc_runner=nvcc))
    ctx = AgentContext(
        run_id=generate_run_id(),
        kernel=kernel,
        arch=settings.compiler.default_arch,
    )

    ctx = await profiler.run(ctx)

    if ctx.has_errors:
        for e in ctx.errors:
            print_error(e)
        return None

    return ctx.profile_metrics


def _print_metrics(metrics) -> None:
    """Print profiling metrics in a readable format."""
    lines = []
    if metrics.gpu_time_us is not None:
        lines.append(f"  GPU Time:          {metrics.gpu_time_us:.2f} µs")
    if metrics.sm_throughput is not None:
        lines.append(f"  SM Throughput:     {metrics.sm_throughput:.1f}%")
    if metrics.dram_throughput is not None:
        lines.append(f"  DRAM Throughput:   {metrics.dram_throughput:.1f}%")
    if metrics.achieved_occupancy is not None:
        lines.append(f"  Occupancy:         {metrics.achieved_occupancy:.1%}")
    if metrics.registers_per_thread is not None:
        lines.append(f"  Registers/Thread:  {metrics.registers_per_thread}")
    if metrics.block_size:
        lines.append(f"  Block Size:        {tuple(metrics.block_size)}")
    if metrics.grid_size:
        lines.append(f"  Grid Size:         {tuple(metrics.grid_size)}")

    for line in lines:
        print_info(line)
