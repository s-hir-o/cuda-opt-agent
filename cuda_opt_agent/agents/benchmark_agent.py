"""BenchmarkAgent — compile, check correctness, benchmark, and compare.

Handles the full evaluation pipeline for a kernel variant:
1. Compile the transformed source to a runnable executable
2. Check correctness against baseline (MANDATORY — fail closed)
3. Benchmark execution time
4. Compare with baseline speedup

Compilation failures do NOT crash the pipeline — they are recorded
and the variant is skipped.
"""

from __future__ import annotations

from cuda_opt_agent.agents.base import AgentContext
from cuda_opt_agent.benchmark.comparator import compare_variants
from cuda_opt_agent.benchmark.runner import BenchmarkRunner
from cuda_opt_agent.build.nvcc_runner import NVCCRunner
from cuda_opt_agent.core.config import BenchmarkConfig
from cuda_opt_agent.core.logging import get_logger
from cuda_opt_agent.schemas.benchmark import BenchmarkResult, SpeedupResult
from cuda_opt_agent.schemas.compilation import CompilationResult
from cuda_opt_agent.schemas.correctness import CorrectnessResult
from cuda_opt_agent.schemas.kernel import KernelSource, KernelVariant

log = get_logger(__name__)


class BenchmarkAgent:
    """Compile, verify correctness, benchmark, and compute speedup."""

    def __init__(
        self,
        nvcc_runner: NVCCRunner,
        benchmark_runner: BenchmarkRunner,
        benchmark_config: BenchmarkConfig,
    ) -> None:
        self._nvcc = nvcc_runner
        self._bench = benchmark_runner
        self._config = benchmark_config

    async def compile(
        self,
        kernel: KernelSource | KernelVariant,
        arch: str,
    ) -> CompilationResult:
        """Compile kernel to a runnable executable for benchmarking.

        Uses compile_executable() so the output is a host-runnable binary,
        not a device-only cubin. Sets CompilationResult.binary_path on success.
        """
        return await self._nvcc.compile_executable(kernel, arch=arch)

    async def check_correctness(
        self,
        variant_id: str,
        baseline_id: str,
        variant_result: CompilationResult,
        baseline_result: CompilationResult | None,
    ) -> CorrectnessResult:
        """Check correctness of variant against baseline.

        Fails closed: returns passed=False unless a real comparison is performed.
        If comparison infrastructure is not available, the variant is blocked
        from benchmarking with a diagnostic message.

        Args:
            variant_id: ID of the variant being checked.
            baseline_id: ID of the baseline variant.
            variant_result: Compilation result of the variant.
            baseline_result: Compilation result of the baseline (None if unavailable).
        """
        if not variant_result.success:
            return CorrectnessResult(
                passed=False,
                variant_id=variant_id,
                baseline_variant_id=baseline_id,
                message="Compilation failed — cannot check correctness",
            )

        # Fail closed: no baseline to compare against
        if baseline_result is None:
            return CorrectnessResult(
                passed=False,
                variant_id=variant_id,
                baseline_variant_id=baseline_id,
                message=(
                    "No baseline compilation result available for correctness comparison. "
                    "Cannot verify correctness — blocking benchmark."
                ),
            )

        if not baseline_result.success:
            return CorrectnessResult(
                passed=False,
                variant_id=variant_id,
                baseline_variant_id=baseline_id,
                message="Baseline compilation failed — cannot compare",
            )

        # TODO: Real GPU comparison — run both kernels, compare output arrays
        # element-wise against tolerance. This requires GPU runtime integration
        # which is beyond the scope of the agent layer. For now, we require
        # a functional comparison to be injected or implemented in the runtime.
        # Until then, fail closed.
        return CorrectnessResult(
            passed=False,
            variant_id=variant_id,
            baseline_variant_id=baseline_id,
            message=(
                "GPU output comparison not yet implemented. "
                "Correctness cannot be verified — blocking benchmark. "
                "Enable GPU runtime comparison to allow benchmarking."
            ),
        )

    def get_executable_path(self, result: CompilationResult) -> str | None:
        """Extract the runnable binary path from a compilation result.

        Returns binary_path if set (from compile_executable), otherwise
        falls back to cubin_path for backward compatibility.

        Returns None if no runnable binary was produced.
        """
        if result.binary_path:
            return result.binary_path
        if result.cubin_path:
            return result.cubin_path
        return None

    async def benchmark_variant(
        self,
        binary_path: str,
        variant_id: str,
        arch: str,
        block_size: tuple[int, ...] | None = None,
        grid_size: tuple[int, ...] | None = None,
        correct: bool = True,
    ) -> BenchmarkResult:
        """Benchmark a compiled kernel variant."""
        return await self._bench.benchmark(
            binary_path=binary_path,
            variant_id=variant_id,
            warmup_iterations=self._config.warmup_iterations,
            timing_iterations=self._config.timing_iterations,
            arch=arch,
            block_size=block_size,
            grid_size=grid_size,
            correct=correct,
        )

    async def evaluate_variant(
        self,
        ctx: AgentContext,
        variant: KernelVariant,
        baseline_benchmark: BenchmarkResult | None,
        baseline_variant_id: str,
        baseline_compilation: CompilationResult | None = None,
    ) -> AgentContext:
        """Full evaluation: compile -> correctness -> benchmark -> speedup.

        Returns a NEW AgentContext for this variant (does not modify the original).
        """
        variant_ctx = AgentContext(
            run_id=ctx.run_id,
            kernel=variant,
            arch=ctx.arch,
        )

        # ── Compile ────────────────────────────────────────────
        log.info("benchmark_agent.compile", variant_id=variant.variant_id)
        try:
            compilation = await self.compile(variant, ctx.arch)
            variant_ctx.compilation = compilation

            if not compilation.success:
                err_msgs = [d.message for d in compilation.diagnostics[:3]]
                variant_ctx.add_error(f"Compilation failed: {'; '.join(err_msgs)}")
                return variant_ctx
        except Exception as exc:
            variant_ctx.add_error(f"Compilation exception: {exc}")
            return variant_ctx

        # ── Correctness (fail closed) ──────────────────────────
        log.info("benchmark_agent.correctness", variant_id=variant.variant_id)
        correctness = await self.check_correctness(
            variant_id=variant.variant_id,
            baseline_id=baseline_variant_id,
            variant_result=compilation,
            baseline_result=baseline_compilation,
        )
        variant_ctx.correctness = correctness

        if not correctness.passed:
            variant_ctx.add_error(f"Correctness check failed: {correctness.message}")
            return variant_ctx

        # ── Extract executable path ────────────────────────────
        exe_path = self.get_executable_path(compilation)
        if exe_path is None:
            variant_ctx.add_error(
                "No runnable binary produced by compilation — cannot benchmark"
            )
            return variant_ctx

        # ── Benchmark ──────────────────────────────────────────
        log.info("benchmark_agent.benchmark", variant_id=variant.variant_id)
        try:
            bench_result = await self.benchmark_variant(
                binary_path=exe_path,
                variant_id=variant.variant_id,
                arch=ctx.arch,
                correct=correctness.passed,
            )
            variant_ctx.benchmark = bench_result
        except Exception as exc:
            variant_ctx.add_error(f"Benchmark failed: {exc}")
            return variant_ctx

        # ── Speedup ────────────────────────────────────────────
        if baseline_benchmark is not None:
            speedup = compare_variants(baseline_benchmark, bench_result)
            variant_ctx.speedup = speedup
            log.info(
                "benchmark_agent.speedup",
                variant_id=variant.variant_id,
                speedup=round(speedup.speedup, 3),
                significant=speedup.significant,
            )

        return variant_ctx
