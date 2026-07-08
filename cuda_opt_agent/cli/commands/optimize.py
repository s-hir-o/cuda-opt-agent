"""`cuda-opt optimize` command — run the full optimization pipeline."""

from __future__ import annotations

import asyncio
from pathlib import Path

import typer

from cuda_opt_agent.cli.output import print_error, print_info, print_run_summary, print_iteration_table, print_success
from cuda_opt_agent.core.config import load_settings
from cuda_opt_agent.schemas.kernel import KernelLanguage, KernelSource


def optimize_cmd(
    file: Path = typer.Argument(..., help="Path to CUDA source file (.cu)"),
    arch: str = typer.Option(None, "--arch", "-a", help="Target GPU architecture (e.g., sm_80)"),
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

    settings = load_settings(config)
    if arch:
        settings.compiler.default_arch = arch
    settings.optimization.max_iterations = max_iterations

    kernel = KernelSource(
        name=file.name,
        language=KernelLanguage.CUDA,
        source_code=source_code,
        entry_point="main",
        file_path=str(file),
    )

    print_info(f"Optimizing {file.name} (arch={settings.compiler.default_arch}, max_iter={max_iterations})")

    result = asyncio.run(_run_optimize(kernel, settings))

    if result is None:
        print_error("Optimization failed — see logs for details")
        raise typer.Exit(code=1)

    print_success(f"\nOptimization complete: {result.status}")
    print_run_summary(result)
    print_iteration_table(result)


async def _run_optimize(kernel: KernelSource, settings):
    """Build coordinator and run optimization."""
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

    # Init DB for persistence
    from cuda_opt_agent.memory.database import init_db
    init_db(settings.database)

    nvcc = NVCCRunner(settings.compiler)

    # LLM — optional, graceful fallback
    try:
        llm = create_provider(settings.llm)
    except Exception:
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

    transform_agent = TransformAgent(llm) if llm else TransformAgent(_FallbackLLM())

    coordinator = Coordinator(
        profiler_agent=ProfilerAgent(MockProfiler()),
        analysis_agent=AnalysisAgent(
            analyzer=MockPTXAnalyzer(),
            feature_pipeline=FeaturePipeline([NCUFeatureExtractor()]),
            cost_model=RooflineModel(),
        ),
        bottleneck_agent=BottleneckAgent(),
        transform_agent=transform_agent,
        benchmark_agent=BenchmarkAgent(
            nvcc_runner=nvcc,
            benchmark_runner=MockBenchmarkRunner(),
            benchmark_config=settings.benchmark,
        ),
        search_strategy=GridSearch(batch_size=settings.search.grid.parallelism),
        config=settings,
    )

    try:
        return await coordinator.optimize(kernel)
    except Exception as exc:
        print_error(f"Optimization crashed: {exc}")
        return None
