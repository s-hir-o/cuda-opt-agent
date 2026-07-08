"""Abstract benchmark runner interface.

Orchestrates: validation → warmup iterations → timing iterations → stats.
Returns a single BenchmarkResult per kernel invocation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from cuda_opt_agent.benchmark.stats import compute_timing_stats, is_stable
from cuda_opt_agent.core.exceptions import BenchmarkError
from cuda_opt_agent.schemas.benchmark import BenchmarkResult, MeasurementStrategy


class BenchmarkRunner(ABC):
    """Time a compiled CUDA kernel execution.

    Subclasses implement run_kernel() with the actual binary invocation
    and set MEASUREMENT_STRATEGY to identify their timing method.
    The base class handles validation, warmup/timing loop, and stat aggregation.
    """

    MEASUREMENT_STRATEGY: MeasurementStrategy | str = ""

    @abstractmethod
    async def run_kernel(
        self,
        binary_path: str,
        kernel_name: str | None = None,
        timeout_s: float = 120.0,
    ) -> list[float]:
        """Execute the kernel once and return elapsed time(s) in milliseconds.

        Implementations may return a single-element list (host timer) or
        multiple elements if measuring sub-kernel phases.

        Raises BenchmarkError on execution failure.
        """

    async def benchmark(
        self,
        binary_path: str,
        variant_id: str = "",
        kernel_name: str | None = None,
        warmup_iterations: int = 3,
        timing_iterations: int = 10,
        timeout_s: float = 120.0,
        arch: str = "",
        block_size: tuple[int, ...] | None = None,
        grid_size: tuple[int, ...] | None = None,
        correct: bool = True,
        stability_cv_threshold: float | None = None,
    ) -> BenchmarkResult:
        """Full benchmark pipeline: validate → warmup → measure → compute stats.

        Args:
            binary_path: Path to compiled executable.
            variant_id: Identifier for this kernel variant.
            kernel_name: Optional kernel name filter.
            warmup_iterations: Warmup runs (not timed).
            timing_iterations: Timed runs for statistics (must be >= 5).
            timeout_s: Per-invocation timeout.
            arch: GPU architecture string.
            block_size: Kernel block dimensions.
            grid_size: Kernel grid dimensions.
            correct: Whether this variant passed correctness check.
            stability_cv_threshold: If set, raise BenchmarkError when CV
                (coefficient of variation) exceeds this threshold.

        Returns:
            BenchmarkResult with TimingStats and metadata.

        Raises:
            BenchmarkError: If timing_iterations < 5, if a warmup or timing
                run fails, if a timing iteration returns no data, or if
                stability_cv_threshold is set and exceeded.
        """
        if timing_iterations < 5:
            raise BenchmarkError(
                f"timing_iterations must be >= 5 for meaningful statistics, "
                f"got {timing_iterations}"
            )

        # Warmup — discard results but propagate failures
        for _ in range(warmup_iterations):
            await self.run_kernel(binary_path, kernel_name, timeout_s)

        # Timed runs — every iteration must succeed
        raw_times: list[float] = []
        for i in range(timing_iterations):
            times = await self.run_kernel(binary_path, kernel_name, timeout_s)
            if not times:
                raise BenchmarkError(
                    f"Timing iteration {i + 1}/{timing_iterations} returned "
                    f"no data for {binary_path}"
                )
            raw_times.append(sum(times))

        timing = compute_timing_stats(raw_times, remove_outliers=False)

        # Optional stability check
        if stability_cv_threshold is not None and not is_stable(
            timing, cv_threshold=stability_cv_threshold
        ):
            from cuda_opt_agent.benchmark.stats import coefficient_of_variation

            cv = coefficient_of_variation(timing)
            raise BenchmarkError(
                f"Benchmark for {variant_id or binary_path} is unstable: "
                f"CV={cv:.4f} exceeds threshold={stability_cv_threshold}"
            )

        return BenchmarkResult(
            variant_id=variant_id,
            timing=timing,
            arch=arch,
            block_size=block_size,
            grid_size=grid_size,
            correct=correct,
            measurement_strategy=self.MEASUREMENT_STRATEGY,
        )

    @abstractmethod
    async def is_available(self) -> bool:
        """Return True if the benchmark infrastructure is ready."""
