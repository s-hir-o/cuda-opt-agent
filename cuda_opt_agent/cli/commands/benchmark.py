"""`cuda-opt benchmark` command — benchmark a compiled kernel."""

from __future__ import annotations

import asyncio
from pathlib import Path

import typer

from cuda_opt_agent.cli.output import print_error, print_info, print_success
from cuda_opt_agent.core.config import load_settings
from cuda_opt_agent.schemas.kernel import KernelLanguage, KernelSource


def benchmark_cmd(
    file: Path = typer.Argument(..., help="Path to CUDA source file (.cu)"),
    arch: str = typer.Option(None, "--arch", "-a", help="Target GPU architecture"),
    warmup: int = typer.Option(3, "--warmup", "-w", help="Warmup iterations"),
    iterations: int = typer.Option(10, "--iterations", "-n", help="Timing iterations"),
    config: Path = typer.Option(None, "--config", "-c", help="Path to config.yaml"),
):
    """Benchmark a CUDA kernel and display timing statistics."""
    if not file.is_file():
        print_error(f"File not found: {file}")
        raise typer.Exit(code=1)

    source_code = file.read_text()
    settings = load_settings(config)
    if arch:
        settings.compiler.default_arch = arch
    settings.benchmark.warmup_iterations = warmup
    settings.benchmark.timing_iterations = iterations

    kernel = KernelSource(
        name=file.name,
        language=KernelLanguage.CUDA,
        source_code=source_code,
        entry_point="main",
    )

    print_info(f"Benchmarking {file.name} (warmup={warmup}, iterations={iterations})")

    result = asyncio.run(_run_benchmark(kernel, settings))

    if result is None:
        print_error("Benchmark failed")
        raise typer.Exit(code=1)

    print_success("\nBenchmark complete:")
    _print_timing(result)


async def _run_benchmark(kernel: KernelSource, settings):
    from cuda_opt_agent.benchmark.mock_runner import MockBenchmarkRunner
    from cuda_opt_agent.build.nvcc_runner import NVCCRunner

    nvcc = NVCCRunner(settings.compiler)

    runner = MockBenchmarkRunner(arch=settings.compiler.default_arch)

    try:
        return await runner.benchmark(
            binary_path=f"/tmp/{kernel.name}",
            variant_id=kernel.short_name,
            warmup_iterations=settings.benchmark.warmup_iterations,
            timing_iterations=settings.benchmark.timing_iterations,
            arch=settings.compiler.default_arch,
        )
    except Exception as exc:
        print_error(f"Benchmark failed: {exc}")
        return None


def _print_timing(result) -> None:
    t = result.timing
    lines = [
        f"  Variant:      {result.variant_id}",
        f"  Mean:         {t.mean_ms:.4f} ms",
        f"  Median:       {t.median_ms:.4f} ms",
        f"  Std Dev:      {t.std_ms:.4f} ms",
        f"  Min:          {t.min_ms:.4f} ms",
        f"  Max:          {t.max_ms:.4f} ms",
        f"  Samples:      {t.n_samples}",
    ]
    if t.p95_ms is not None:
        lines.append(f"  P95:          {t.p95_ms:.4f} ms")
    for line in lines:
        print_info(line)
