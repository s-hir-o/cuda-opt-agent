"""`cuda-opt optimize` command — run the full optimization pipeline."""

from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import Literal

import typer

from cuda_opt_agent.cli.output import print_error, print_info, print_run_summary, print_iteration_table, print_success
from cuda_opt_agent.core.config import load_settings
from cuda_opt_agent.schemas.kernel import KernelLanguage, KernelSource
from cuda_opt_agent.schemas.optimization import OptimizationIteration, OptimizationRun

LLMProviderName = Literal["claude", "openai", "gemini", "ollama", "vllm"]


def optimize_cmd(
    file: Path = typer.Argument(..., help="Path to CUDA source file (.cu)"),
    arch: str = typer.Option(None, "--arch", "-a", help="Target GPU architecture (e.g., sm_80)"),
    entry_point: str = typer.Option(
        "main",
        "--entry-point",
        "-e",
        help="CUDA kernel entry point to optimize",
    ),
    output: Path = typer.Option(
        None,
        "--output",
        "-o",
        help="Path where the optimized CUDA source will be written",
    ),
    provider: LLMProviderName | None = typer.Option(
        None,
        "--provider",
        help="LLM provider override: claude, openai, gemini, ollama, or vllm",
    ),
    max_iterations: int = typer.Option(20, "--max-iter", "-n", help="Maximum optimization iterations"),
    config: Path = typer.Option(None, "--config", "-c", help="Path to config.yaml"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
):
    """Run the full optimization pipeline on a CUDA kernel."""
    if not file.is_file():
        print_error(f"File not found: {file}")
        raise typer.Exit(code=1)

    source_code = file.read_text()
    if not source_code.strip():
        print_error(f"File is empty: {file}")
        raise typer.Exit(code=1)
    contract_errors = _validate_runnable_cuda_contract(source_code, entry_point)
    if contract_errors:
        for error in contract_errors:
            print_error(error)
        raise typer.Exit(code=1)

    settings = load_settings(config)
    if arch:
        settings.compiler.default_arch = arch
    if provider:
        settings.llm.provider = provider
    settings.optimization.max_iterations = max_iterations

    kernel = KernelSource(
        name=file.name,
        language=KernelLanguage.CUDA,
        source_code=source_code,
        entry_point=entry_point,
        file_path=str(file),
    )

    output_note = f", output={output}" if output else ""
    print_info(
        f"Optimizing {file.name} "
        f"(entry_point={entry_point}, arch={settings.compiler.default_arch}, "
        f"max_iter={max_iterations}, provider={settings.llm.provider}{output_note})"
    )

    result = asyncio.run(_run_optimize(kernel, settings))

    if result is None:
        print_error("Optimization failed — see logs for details")
        raise typer.Exit(code=1)

    if output and not _write_best_variant_source(result, output):
        raise typer.Exit(code=1)

    print_success(f"\nOptimization complete: {result.status}")
    print_run_summary(result)
    print_iteration_table(result)


def _write_best_variant_source(run: OptimizationRun, output: Path) -> bool:
    """Write the best available variant source to disk."""
    if output.exists() and output.is_dir():
        print_error(f"Output path is a directory: {output}")
        return False

    best_iteration = _find_best_iteration(run)
    if best_iteration is None:
        print_error("No optimization iterations were produced; nothing to write")
        return False

    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(best_iteration.variant.source_code)
    except OSError as exc:
        print_error(f"Failed to write optimized source to {output}: {exc}")
        return False

    if best_iteration.variant_id == "baseline" or best_iteration.validated_speedup <= 1.0:
        print_info("No faster variant was found; wrote the best available source")

    print_success(f"Optimized source written to {output}")
    return True


def _find_best_iteration(run: OptimizationRun) -> OptimizationIteration | None:
    """Return the iteration that should be emitted as the optimized source."""
    if not run.iterations:
        return None

    if run.best_variant_id:
        for iteration in run.iterations:
            if iteration.variant_id == run.best_variant_id:
                return iteration

    return max(run.iterations, key=lambda iteration: iteration.validated_speedup)


def _validate_runnable_cuda_contract(source_code: str, entry_point: str) -> list[str]:
    """Validate the source contract required by the real optimize pipeline."""
    errors: list[str] = []

    if not _declares_global_entry_point(source_code, entry_point):
        errors.append(
            f"CUDA entry point '{entry_point}' was not declared as "
            f"`__global__ void {entry_point}(...)`"
        )

    if not re.search(r"\bint\s+main\s*\(", source_code):
        errors.append(
            "Input must include a host `main()` so nvcc can build a runnable "
            "benchmark executable"
        )

    if '"--output"' not in source_code and "'--output'" not in source_code:
        errors.append(
            "Input executable must support the correctness contract "
            "`--output <path>` and write its raw numeric output buffer there"
        )

    return errors


def _declares_global_entry_point(source_code: str, entry_point: str) -> bool:
    if not entry_point:
        return False
    pattern = re.compile(
        r"__global__\s+void\s+" + re.escape(entry_point) + r"\s*\(",
        re.MULTILINE,
    )
    return pattern.search(source_code) is not None


async def _run_optimize(kernel: KernelSource, settings):
    """Build coordinator and run optimization."""
    from cuda_opt_agent.agents.coordinator import Coordinator
    from cuda_opt_agent.agents.profiler_agent import ProfilerAgent
    from cuda_opt_agent.agents.analysis_agent import AnalysisAgent
    from cuda_opt_agent.agents.bottleneck_agent import BottleneckAgent
    from cuda_opt_agent.agents.transform_agent import TransformAgent
    from cuda_opt_agent.agents.generator_agent import GeneratorAgent
    from cuda_opt_agent.agents.benchmark_agent import BenchmarkAgent
    from cuda_opt_agent.analysis.ptx.mock_analyzer import MockPTXAnalyzer
    from cuda_opt_agent.benchmark.runner import ExecutableBenchmarkRunner
    from cuda_opt_agent.build.nvcc_runner import NVCCRunner
    from cuda_opt_agent.correctness.file_runner import FileOutputCorrectnessRunner
    from cuda_opt_agent.cost_models.analytical.roofline import RooflineModel
    from cuda_opt_agent.features.extractors.ncu_features import NCUFeatureExtractor
    from cuda_opt_agent.features.pipeline import FeaturePipeline
    from cuda_opt_agent.llm.factory import create_provider
    from cuda_opt_agent.profiling.ncu_runner import NCUProfiler, NCURunner
    from cuda_opt_agent.search.grid_search import GridSearch

    # Init DB for persistence
    from cuda_opt_agent.memory.database import init_db
    init_db(settings.database)

    nvcc = NVCCRunner(settings.compiler)
    ncu = NCURunner(settings.profiling)

    # LLM - vLLM is explicit and must be reachable; other providers retain
    # their historical optional fallback behavior.
    try:
        llm = create_provider(settings.llm)
        if settings.llm.provider == "vllm" and not await llm.health_check():
            base_url = settings.llm.vllm.base_url or "http://localhost:8000/v1"
            print_error(
                "vLLM provider is not reachable at "
                f"{base_url}; start the configured vLLM server before optimizing"
            )
            return None
    except Exception as exc:
        if settings.llm.provider == "vllm":
            print_error(f"Failed to initialize vLLM provider: {exc}")
            return None
        llm = None

    class _FallbackLLM:
        def provider_name(self): return "none"
        def estimate_cost(self, i, o): return 0
        async def health_check(self): return False
        async def generate(self, req):
            from cuda_opt_agent.schemas.llm import LLMResponse
            return LLMResponse(content='{"analysis_summary":"no LLM","transforms":[]}', model="none", provider="none")
        async def generate_json(self, req, retries=3):
            return await self.generate(req)

    generator_agent = GeneratorAgent(llm) if llm else None
    transform_agent = TransformAgent(llm) if llm and generator_agent is None else TransformAgent(_FallbackLLM())

    coordinator = Coordinator(
        profiler_agent=ProfilerAgent(NCUProfiler(ncu_runner=ncu, nvcc_runner=nvcc)),
        analysis_agent=AnalysisAgent(
            analyzer=MockPTXAnalyzer(),
            feature_pipeline=FeaturePipeline([NCUFeatureExtractor()]),
            cost_model=RooflineModel(),
            nvcc_runner=nvcc,
        ),
        bottleneck_agent=BottleneckAgent(),
        transform_agent=transform_agent,
        benchmark_agent=BenchmarkAgent(
            nvcc_runner=nvcc,
            benchmark_runner=ExecutableBenchmarkRunner(
                use_cuda_events=settings.benchmark.use_cuda_events,
            ),
            benchmark_config=settings.benchmark,
            correctness_runner=FileOutputCorrectnessRunner(
                timeout_seconds=settings.benchmark.timeout_seconds,
            ),
            correctness_tolerance_absolute=settings.optimization.correctness_tolerance.absolute,
            correctness_tolerance_relative=settings.optimization.correctness_tolerance.relative,
        ),
        search_strategy=GridSearch(batch_size=settings.search.grid.parallelism),
        config=settings,
        generator_agent=generator_agent,
    )

    try:
        return await coordinator.optimize(kernel)
    except Exception as exc:
        print_error(f"Optimization crashed: {exc}")
        return None
