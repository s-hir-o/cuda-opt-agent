"""Mock benchmark runner for testing without a GPU.

Returns deterministic timing data based on kernel name and variant.
Uses SHA-256 for cross-session reproducible seeding (Python's built-in
hash() is randomized per interpreter session via    HASHSEED).

The deterministic benchmark() override ensures timing samples are
generated from a seed derived solely from (binary_path, variant_id),
so warmup count does not affect the timing distribution.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import random

from cuda_opt_agent.benchmark.runner import BenchmarkRunner
from cuda_opt_agent.benchmark.stats import compute_timing_stats
from cuda_opt_agent.core.exceptions import BenchmarkError
from cuda_opt_agent.schemas.benchmark import (
    BenchmarkResult,
    MeasurementStrategy,
)


def _stable_hash(s: str) -> int:
    """Deterministic hash for seeding across     sessions.

    Uses SHA-256 to produce a consistent integer regardless of
       HASHSEED or interpreter version.
    """
    digest = hashlib.sha256(s.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


class MockBenchmarkRunner(BenchmarkRunner):
    """Returns deterministic benchmark results. No GPU required.

    Overrides benchmark() to generate timing samples via a seed derived
    solely from (binary_path, variant_id), ensuring that results are
    identical regardless of warmup_iterations.

    Only implements run_kernel() + benchmark() + is_available().
    The warmup/iteration/validation logic in the base benchmark() is
    exercised by tests that construct a MinimalBenchmarkRunner subclass
    that does NOT override benchmark().
    """

    MEASUREMENT_STRATEGY = MeasurementStrategy.MOCK

    _PRESETS: dict[str, dict] = {
        "naive_matmul": {
            "base_time_ms": 8.5,
            "std_ratio": 0.03,
        },
        "tiled_matmul": {
            "base_time_ms": 1.2,
            "std_ratio": 0.02,
        },
        "vecadd": {
            "base_time_ms": 0.45,
            "std_ratio": 0.05,
        },
    }
    _DEFAULT = {"base_time_ms": 5.0, "std_ratio": 0.04}

    def __init__(
        self,
        arch: str = "sm_80",
        seed: int = 42,
        block_size: tuple[int, ...] | None = None,
        grid_size: tuple[int, ...] | None = None,
    ) -> None:
        self._arch = arch
        self._base_seed = seed
        self._run_counter = 0
        self._block_size = block_size or (16, 16, 1)
        self._grid_size = grid_size or (64, 64, 1)

    async def is_available(self) -> bool:
        return True

    async def run_kernel(
        self,
        binary_path: str,
        kernel_name: str | None = None,
        timeout_s: float = 120.0,
    ) -> list[float]:
        """Return a deterministic fake timing for a single kernel run.

        Uses stable hash of binary_path + monotonically increasing counter.
        Called by the base class when run_kernel is invoked directly.
        """
        name = kernel_name or _name_from_path(binary_path)
        preset = self._PRESETS.get(name, self._DEFAULT)

        base = preset["base_time_ms"]
        std = base * preset["std_ratio"]

        self._run_counter += 1
        seed = _stable_hash(binary_path) + self._base_seed + self._run_counter
        rng = random.Random(seed)
        time_ms = max(0.01, rng.gauss(base, std))

        # Simulate kernel execution delay (scaled down for tests)
        await asyncio.sleep(0.001)

        return [time_ms]

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
        """Deterministic benchmark: seed from (binary_path, variant_id, base_seed).

        Timing samples are generated from a seed that is independent of
        warmup_iterations, so the same configuration always produces
        identical timing data regardless of warmup count.

        Validates timing_iterations >= 5 per the base class contract.
        """
        if timing_iterations < 5:
            raise BenchmarkError(
                f"timing_iterations must be >= 5 for meaningful statistics, "
                f"got {timing_iterations}"
            )

        name = kernel_name or _name_from_path(binary_path)
        preset = self._PRESETS.get(name, self._DEFAULT)

        base = preset["base_time_ms"]
        std = base * preset["std_ratio"]

        # Seed derived from (binary_path, variant_id, base_seed) ONLY.
        # Warmup iterations do not affect the seed.
        seed = _stable_hash(binary_path) + _stable_hash(variant_id) + self._base_seed
        rng = random.Random(seed)
        raw_times = [
            max(0.01, rng.gauss(base, std))
            for _ in range(timing_iterations)
        ]

        timing = compute_timing_stats(raw_times, remove_outliers=False)

        # Optional stability check
        if stability_cv_threshold is not None:
            from cuda_opt_agent.benchmark.stats import coefficient_of_variation, is_stable

            if not is_stable(timing, cv_threshold=stability_cv_threshold):
                cv = coefficient_of_variation(timing)
                raise BenchmarkError(
                    f"Benchmark for {variant_id or binary_path} is unstable: "
                    f"CV={cv:.4f} exceeds threshold={stability_cv_threshold}"
                )

        return BenchmarkResult(
            variant_id=variant_id,
            timing=timing,
            arch=arch or self._arch,
            block_size=block_size or self._block_size,
            grid_size=grid_size or self._grid_size,
            correct=correct,
            measurement_strategy=self.MEASUREMENT_STRATEGY,
        )


def _name_from_path(binary_path: str) -> str:
    """Extract kernel name from binary path."""
    name = os.path.basename(binary_path)
    name = name.split(".")[0]
    for suffix in ("_exec", "_cubin", "_tiled", "_unrolled"):
        name = name.removesuffix(suffix)
    return name
