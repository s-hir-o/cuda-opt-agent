"""Coordinator — the main optimization loop.

Orchestrates all agents in the correct pipeline order:
1. Profile baseline
2. Analyze (PTX, features, cost model)
3. Classify bottleneck
4. Compile + benchmark baseline (establishes reference)
5. Generate transform candidates (search + LLM)
6. Evaluate ALL candidates in each batch
7. Track best variant and convergence
8. Repeat or finish

Compilation failures and incorrect variants are skipped, never crashing.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from cuda_opt_agent.agents.analysis_agent import AnalysisAgent
from cuda_opt_agent.agents.base import AgentContext
from cuda_opt_agent.agents.benchmark_agent import BenchmarkAgent
from cuda_opt_agent.agents.bottleneck_agent import BottleneckAgent
from cuda_opt_agent.agents.profiler_agent import ProfilerAgent
from cuda_opt_agent.agents.transform_agent import TransformAgent
from cuda_opt_agent.core.config import AppSettings
from cuda_opt_agent.core.ids import generate_run_id, generate_variant_id
from cuda_opt_agent.core.logging import get_logger
from cuda_opt_agent.search.base import SearchContext, SearchResult, SearchStrategy
from cuda_opt_agent.schemas.benchmark import BenchmarkResult
from cuda_opt_agent.schemas.compilation import CompilationResult
from cuda_opt_agent.schemas.kernel import KernelSource, KernelVariant
from cuda_opt_agent.schemas.optimization import OptimizationIteration, OptimizationRun
from cuda_opt_agent.schemas.transform import TransformSpec

log = get_logger(__name__)


class Coordinator:
    """Main optimization loop — drives the full pipeline.

    Usage:
        coordinator = Coordinator(...)
        run = await coordinator.optimize(kernel)
    """

    def __init__(
        self,
        profiler_agent: ProfilerAgent,
        analysis_agent: AnalysisAgent,
        bottleneck_agent: BottleneckAgent,
        transform_agent: TransformAgent,
        benchmark_agent: BenchmarkAgent,
        search_strategy: SearchStrategy,
        config: AppSettings,
    ) -> None:
        self._profiler = profiler_agent
        self._analysis = analysis_agent
        self._bottleneck = bottleneck_agent
        self._transform = transform_agent
        self._benchmark = benchmark_agent
        self._search = search_strategy
        self._config = config

    async def optimize(self, kernel: KernelSource) -> OptimizationRun:
        """Run the full optimization pipeline on a kernel.

        Args:
            kernel: The kernel source to optimize.

        Returns:
            OptimizationRun with all iterations and the best variant.
        """
        run_id = generate_run_id()
        arch = self._config.compiler.default_arch
        max_iter = self._config.optimization.max_iterations

        run = OptimizationRun(
            run_id=run_id,
            kernel=kernel,
            target_arch=arch,
            max_iterations=max_iter,
            config_snapshot=self._config.model_dump(mode="json"),
        )

        log.info("coordinator.start", run_id=run_id, kernel=kernel.name, arch=arch)

        # ── Step 1: Profile, analyze, classify baseline ────────
        baseline_ctx = await self._analyze_baseline(kernel, run_id, arch)

        # ── Step 2: Compile and benchmark the baseline ─────────
        baseline_benchmark, baseline_compilation = (
            await self._benchmark_baseline(kernel, run_id, arch, baseline_ctx)
        )

        # Record baseline iteration
        baseline_variant = _make_variant(kernel, "baseline", 0)
        baseline_iter = self._build_iteration(
            iteration_num=0,
            variant_id="baseline",
            variant=baseline_variant,
            ctx=baseline_ctx,
            benchmark_override=baseline_benchmark,
        )
        run.add_iteration(baseline_iter)

        if baseline_benchmark is not None:
            log.info(
                "coordinator.baseline_benchmarked",
                run_id=run_id,
                mean_ms=round(baseline_benchmark.timing.mean_ms, 3),
            )
        else:
            log.warning("coordinator.no_baseline_benchmark", run_id=run_id)

        # ── Step 3: Optimization loop ──────────────────────────
        best_speedup = 1.0
        no_improvement_count = 0
        patience = self._config.optimization.convergence_patience
        iteration = 0

        self._search.reset()
        search_ctx = SearchContext(
            transforms=self._get_applicable_transforms(baseline_ctx),
            iteration=0,
            best_speedup=1.0,
        )

        while iteration < max_iter:
            # Generate candidates
            candidates = self._search.generate_candidates(search_ctx)
            if not candidates:
                # Try LLM if search exhausted
                llm_specs = await self._transform.run(baseline_ctx, max_suggestions=3)
                candidates = llm_specs

            if not candidates:
                log.info("coordinator.no_candidates", run_id=run_id)
                break

            # Evaluate ALL candidates in the batch
            for spec in candidates:
                if iteration >= max_iter:
                    break

                iteration += 1
                log.info(
                    "coordinator.iteration",
                    run_id=run_id,
                    iteration=iteration,
                    transform=spec.transform_name,
                )

                iter_result, search_result = await self._evaluate_and_report(
                    spec=spec,
                    iteration=iteration,
                    kernel=kernel,
                    run_id=run_id,
                    arch=arch,
                    baseline_ctx=baseline_ctx,
                    baseline_benchmark=baseline_benchmark,
                    baseline_compilation=baseline_compilation,
                )

                run.add_iteration(iter_result)

                # Track convergence
                if (
                    iter_result.achieved_speedup > best_speedup * self._config.optimization.min_speedup_threshold
                ):
                    best_speedup = iter_result.achieved_speedup
                    run.best_speedup = best_speedup
                    run.best_variant_id = iter_result.variant_id
                    no_improvement_count = 0
                    search_ctx.best_speedup = best_speedup
                    log.info(
                        "coordinator.new_best",
                        run_id=run_id,
                        speedup=round(best_speedup, 3),
                        variant=iter_result.variant_id,
                    )
                else:
                    no_improvement_count += 1

                search_ctx.iteration = iteration
                search_ctx.previous_results.append(search_result)

                if no_improvement_count >= patience:
                    log.info(
                        "coordinator.converged",
                        run_id=run_id,
                        patience=patience,
                    )
                    run.status = "converged"
                    break

            if run.status == "converged":
                break

        # ── Finalize ───────────────────────────────────────────
        run.completed_at = datetime.now(timezone.utc)
        if run.status == "running":
            run.status = "completed"

        log.info(
            "coordinator.done",
            run_id=run_id,
            iterations=run.iteration_count,
            best_speedup=round(run.best_speedup, 3),
            best_variant=run.best_variant_id,
            status=run.status,
        )

        return run

    # ── Internal pipeline stages ───────────────────────────────

    async def _analyze_baseline(
        self, kernel: KernelSource, run_id: str, arch: str
    ) -> AgentContext:
        """Profile and analyze the baseline kernel (no benchmarking)."""
        ctx = AgentContext(run_id=run_id, kernel=kernel, arch=arch)

        # Profile
        ctx = await self._profiler.run(ctx)

        # Analyze + features + cost model
        ctx = await self._analysis.run(ctx)

        # Classify bottleneck
        ctx = self._bottleneck.run(ctx)

        return ctx

    async def _benchmark_baseline(
        self,
        kernel: KernelSource,
        run_id: str,
        arch: str,
        baseline_ctx: AgentContext,
    ) -> tuple[BenchmarkResult | None, CompilationResult | None]:
        """Compile and benchmark the original kernel to establish a reference.

        Returns:
            Tuple of (baseline_benchmark, baseline_compilation).
            Both are None if any step fails.
        """
        compile_ctx = AgentContext(run_id=run_id, kernel=kernel, arch=arch)

        try:
            log.info("coordinator.baseline_compile", run_id=run_id)
            compilation = await self._benchmark.compile(kernel, arch)
            compile_ctx.compilation = compilation

            if not compilation.success:
                log.warning(
                    "coordinator.baseline_compile_failed",
                    run_id=run_id,
                    errors=[d.message for d in compilation.diagnostics[:3]],
                )
                return None, None

            # Correctness check on baseline (vs itself — should pass if comparison exists)
            correctness = await self._benchmark.check_correctness(
                variant_id="baseline",
                baseline_id="baseline",
                variant_result=compilation,
                baseline_result=compilation,
            )

            if not correctness.passed:
                log.warning(
                    "coordinator.baseline_correctness_blocked",
                    run_id=run_id,
                    message=correctness.message,
                )
                # Baseline correctness failure blocks benchmarking
                compile_ctx.correctness = correctness
                return None, compilation

            compile_ctx.correctness = correctness

            # Extract executable path
            exe_path = self._benchmark.get_executable_path(compilation)
            if exe_path is None:
                log.warning("coordinator.baseline_no_executable", run_id=run_id)
                return None, compilation

            # Benchmark
            benchmark = await self._benchmark.benchmark_variant(
                binary_path=exe_path,
                variant_id="baseline",
                arch=arch,
                correct=correctness.passed,
            )

            return benchmark, compilation

        except Exception as exc:
            log.warning(
                "coordinator.baseline_benchmark_error",
                run_id=run_id,
                error=str(exc),
            )
            return None, None

    async def _evaluate_and_report(
        self,
        spec: TransformSpec,
        iteration: int,
        kernel: KernelSource,
        run_id: str,
        arch: str,
        baseline_ctx: AgentContext,
        baseline_benchmark: BenchmarkResult | None,
        baseline_compilation: CompilationResult | None,
    ) -> tuple[OptimizationIteration, SearchResult]:
        """Evaluate a single candidate and report to the search strategy.

        Returns:
            Tuple of (OptimizationIteration, SearchResult).
        """
        # Apply transform
        variant_id = generate_variant_id(
            kernel.short_name, iteration, spec.transform_name
        )

        try:
            from cuda_opt_agent.transforms.registry import get_transform

            transform = get_transform(spec.transform_name)
            result = transform.apply(kernel, spec.params)

            if not result.success:
                iter_result = self._failed_iteration(
                    iteration, variant_id, kernel,
                    f"Transform {spec.transform_name} failed: {result.error_message}",
                )
                self._report_search_result(spec, iter_result)
                return iter_result, self._make_search_result(spec, iter_result)

            variant = KernelVariant(
                variant_id=variant_id,
                parent_source=kernel,
                source_code=result.transformed_source,
                iteration=iteration,
                transform_history=list(
                    getattr(kernel, "transform_history", [])
                ) + [spec],
            )
        except Exception as exc:
            iter_result = self._failed_iteration(
                iteration, variant_id, kernel, f"Transform error: {exc}"
            )
            self._report_search_result(spec, iter_result)
            return iter_result, self._make_search_result(spec, iter_result)

        # Evaluate variant
        variant_ctx = await self._benchmark.evaluate_variant(
            ctx=AgentContext(run_id=run_id, kernel=kernel, arch=arch),
            variant=variant,
            baseline_benchmark=baseline_benchmark,
            baseline_variant_id="baseline",
            baseline_compilation=baseline_compilation,
        )

        # Log speedup if available
        if variant_ctx.speedup is not None:
            log.info(
                "coordinator.candidate_speedup",
                run_id=run_id,
                variant_id=variant_id,
                speedup=round(variant_ctx.speedup.speedup, 3),
            )

        # Build iteration
        iter_result = self._build_iteration(
            iteration_num=iteration,
            variant_id=variant_id,
            variant=variant,
            ctx=variant_ctx,
        )

        # Report to search
        self._report_search_result(spec, iter_result)
        search_result = self._make_search_result(spec, iter_result)

        return iter_result, search_result

    def _build_iteration(
        self,
        iteration_num: int,
        variant_id: str,
        variant: KernelVariant,
        ctx: AgentContext,
        benchmark_override: BenchmarkResult | None = None,
    ) -> OptimizationIteration:
        """Assemble an OptimizationIteration from an AgentContext."""
        return OptimizationIteration(
            iteration=iteration_num,
            variant_id=variant_id,
            variant=variant,
            profile_metrics=ctx.profile_metrics,
            bottleneck=ctx.bottleneck,
            compilation=ctx.compilation,
            correctness=ctx.correctness,
            benchmark=benchmark_override or ctx.benchmark,
            speedup=ctx.speedup,
            analysis=ctx.analysis,
            features=ctx.features,
            cost_estimate=ctx.cost_estimate,
            completed_at=datetime.now(timezone.utc),
            error="; ".join(ctx.errors) if ctx.errors else None,
        )

    def _failed_iteration(
        self,
        iteration: int,
        variant_id: str,
        kernel: KernelSource,
        error: str,
    ) -> OptimizationIteration:
        """Build an OptimizationIteration for a failed evaluation."""
        return OptimizationIteration(
            iteration=iteration,
            variant_id=variant_id,
            variant=_make_variant(kernel, variant_id, iteration),
            completed_at=datetime.now(timezone.utc),
            error=error,
        )

    def _report_search_result(
        self, spec: TransformSpec, iter_result: OptimizationIteration
    ) -> None:
        """Report evaluation outcome to the search strategy."""
        self._search.report_result(self._make_search_result(spec, iter_result))

    def _make_search_result(
        self, spec: TransformSpec, iter_result: OptimizationIteration
    ) -> SearchResult:
        """Build a SearchResult from the actual iteration outcome."""
        return SearchResult(
            spec=spec,
            speedup=iter_result.achieved_speedup,
            correct=(
                iter_result.correctness.passed
                if iter_result.correctness is not None
                else False
            ),
            benchmark_time_ms=(
                iter_result.benchmark.timing.mean_ms
                if iter_result.benchmark is not None
                else None
            ),
            error=iter_result.error,
        )

    def _get_applicable_transforms(self, ctx: AgentContext) -> list:
        """Get transforms applicable to the baseline bottleneck."""
        if ctx.bottleneck is None:
            from cuda_opt_agent.transforms.registry import list_transforms
            from cuda_opt_agent.transforms.registry import get_transform

            return [get_transform(n) for n in list_transforms()]

        from cuda_opt_agent.transforms.registry import transforms_for_bottleneck

        return transforms_for_bottleneck(ctx.bottleneck.primary)


def _make_variant(
    kernel: KernelSource, variant_id: str, iteration: int
) -> KernelVariant:
    """Create a minimal KernelVariant from a KernelSource."""
    return KernelVariant(
        variant_id=variant_id,
        parent_source=kernel,
        source_code=kernel.source_code,
        iteration=iteration,
    )
