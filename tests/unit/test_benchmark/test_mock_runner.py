"""Tests for MockBenchmarkRunner.

Covers:
- Stable hashing (Issue 1 from prior pass)
- Inherited benchmark loop (Issue 2)
- Minimum timing iterations enforcement (Issue 3)
- Measurement strategy propagation (Issue 5)
- Warmup determinism (Critical Fix #3)
- Stability validation (Critical Fix #5)
- Partial failure rejection (Critical Fix #1)
"""

import pytest

from cuda_opt_agent.core.exceptions import BenchmarkError
from cuda_opt_agent.benchmark.mock_runner import MockBenchmarkRunner, _stable_hash
from cuda_opt_agent.benchmark.runner import BenchmarkRunner
from cuda_opt_agent.schemas.benchmark import MeasurementStrategy


# ── Minimal subclass that does NOT override benchmark() ────────
# Used to test the base class warmup/timing/validation loop.

class MinimalBenchmarkRunner(BenchmarkRunner):
    """Minimal implementation using only run_kernel() from MockBenchmarkRunner.

    Inherits benchmark() from BenchmarkRunner to exercise the full pipeline.
    """

    MEASUREMENT_STRATEGY = MeasurementStrategy.MOCK

    def __init__(self) -> None:
        self._mock = MockBenchmarkRunner()
        self.call_count = 0

    async def run_kernel(self, binary_path, kernel_name=None, timeout_s=120.0):
        self.call_count += 1
        return await self._mock.run_kernel(binary_path, kernel_name, timeout_s)

    async def is_available(self):
        return True


class FailingRunKernelRunner(BenchmarkRunner):
    """Runner that returns empty list (simulating partial failure)."""

    MEASUREMENT_STRATEGY = MeasurementStrategy.MOCK

    call_count: int = 0

    async def run_kernel(self, binary_path, kernel_name=None, timeout_s=120.0):
        FailingRunKernelRunner.call_count += 1
        return []  # No data returned

    async def is_available(self):
        return True


# ── Stable Hashing ─────────────────────────────────────────────

class TestStableHash:

    def test_deterministic(self):
        assert _stable_hash("hello") == _stable_hash("hello")
        assert _stable_hash("/tmp/test_kernel") == _stable_hash("/tmp/test_kernel")

    def test_different_inputs_differ(self):
        assert _stable_hash("hello") != _stable_hash("world")
        assert _stable_hash("/tmp/a") != _stable_hash("/tmp/b")

    def test_returns_int(self):
        result = _stable_hash("test")
        assert isinstance(result, int)
        assert result >= 0

    def test_empty_string(self):
        assert _stable_hash("") == _stable_hash("")
        assert isinstance(_stable_hash(""), int)


# ── Mock Runner Core ──────────────────────────────────────────

class TestMockBenchmarkRunner:

    @pytest.mark.asyncio
    async def test_is_available(self):
        runner = MockBenchmarkRunner()
        assert await runner.is_available() is True

    @pytest.mark.asyncio
    async def test_run_kernel_returns_time(self):
        runner = MockBenchmarkRunner()
        times = await runner.run_kernel("/tmp/naive_matmul")
        assert len(times) == 1
        assert times[0] > 0

    @pytest.mark.asyncio
    async def test_run_kernel_deterministic_across_instances(self):
        r1 = MockBenchmarkRunner(seed=42)
        r2 = MockBenchmarkRunner(seed=42)

        t1 = await r1.run_kernel("/tmp/test")
        t2 = await r2.run_kernel("/tmp/test")
        assert t1 == t2

    @pytest.mark.asyncio
    async def test_run_kernel_sequential_calls_vary(self):
        runner = MockBenchmarkRunner(seed=42)
        t1 = await runner.run_kernel("/tmp/test")
        t2 = await runner.run_kernel("/tmp/test")
        assert t1 != t2

    @pytest.mark.asyncio
    async def test_benchmark_returns_result(self):
        runner = MockBenchmarkRunner()
        result = await runner.benchmark(
            binary_path="/tmp/naive_matmul",
            variant_id="v0",
            warmup_iterations=2,
            timing_iterations=10,
            arch="sm_80",
        )
        assert result.variant_id == "v0"
        assert result.timing.n_samples == 10
        assert result.timing.mean_ms > 0
        assert result.timing.std_ms >= 0
        assert result.arch == "sm_80"

    @pytest.mark.asyncio
    async def test_deterministic_results(self):
        """Same inputs produce same outputs across runner instances."""
        r1 = MockBenchmarkRunner(seed=42)
        r2 = MockBenchmarkRunner(seed=42)

        b1 = await r1.benchmark(
            "/tmp/test", variant_id="v1", timing_iterations=10, arch="sm_80"
        )
        b2 = await r2.benchmark(
            "/tmp/test", variant_id="v1", timing_iterations=10, arch="sm_80"
        )

        assert b1.timing.mean_ms == b2.timing.mean_ms
        assert b1.timing.std_ms == b2.timing.std_ms

    @pytest.mark.asyncio
    async def test_different_kernels_different_times(self):
        r1 = MockBenchmarkRunner()
        naive = await r1.benchmark(
            "/tmp/naive_matmul", variant_id="v0", timing_iterations=10, arch="sm_80"
        )
        r2 = MockBenchmarkRunner()
        tiled = await r2.benchmark(
            "/tmp/tiled_matmul", variant_id="v1", timing_iterations=10, arch="sm_80"
        )
        assert tiled.timing.mean_ms < naive.timing.mean_ms

    @pytest.mark.asyncio
    async def test_custom_arch(self):
        runner = MockBenchmarkRunner()
        result = await runner.benchmark(
            "/tmp/test", variant_id="v1", timing_iterations=10, arch="sm_90"
        )
        assert result.arch == "sm_90"

    @pytest.mark.asyncio
    async def test_correctness_flag_passed_through(self):
        runner = MockBenchmarkRunner()
        result = await runner.benchmark(
            "/tmp/test", variant_id="v1", timing_iterations=10, correct=False
        )
        assert result.correct is False

    @pytest.mark.asyncio
    async def test_launch_config_preserved(self):
        runner = MockBenchmarkRunner()
        result = await runner.benchmark(
            "/tmp/test",
            variant_id="v1",
            timing_iterations=10,
            arch="sm_80",
            block_size=(32, 32, 1),
            grid_size=(128, 128, 1),
        )
        assert result.block_size == (32, 32, 1)
        assert result.grid_size == (128, 128, 1)

    @pytest.mark.asyncio
    async def test_raw_times_populated(self):
        runner = MockBenchmarkRunner()
        result = await runner.benchmark(
            "/tmp/test", variant_id="v1", timing_iterations=10, arch="sm_80"
        )
        assert len(result.timing.raw_times_ms) == 10
        assert all(t > 0 for t in result.timing.raw_times_ms)


# ── Warmup Determinism (Critical Fix #3) ──────────────────────

class TestWarmupDeterminism:

    @pytest.mark.asyncio
    async def test_different_warmup_same_timing(self):
        """Timing samples must be identical regardless of warmup count."""
        r1 = MockBenchmarkRunner(seed=42)
        r2 = MockBenchmarkRunner(seed=42)

        b1 = await r1.benchmark(
            "/tmp/test", variant_id="v1",
            warmup_iterations=0, timing_iterations=10, arch="sm_80",
        )
        b2 = await r2.benchmark(
            "/tmp/test", variant_id="v1",
            warmup_iterations=10, timing_iterations=10, arch="sm_80",
        )

        assert b1.timing.raw_times_ms == b2.timing.raw_times_ms
        assert b1.timing.mean_ms == b2.timing.mean_ms

    @pytest.mark.asyncio
    async def test_warmup_0_vs_warmup_100_same_result(self):
        """Extreme warmup count difference produces same timing."""
        r1 = MockBenchmarkRunner(seed=99)
        r2 = MockBenchmarkRunner(seed=99)

        b1 = await r1.benchmark(
            "/tmp/naive_matmul", variant_id="extreme",
            warmup_iterations=0, timing_iterations=20, arch="sm_80",
        )
        b2 = await r2.benchmark(
            "/tmp/naive_matmul", variant_id="extreme",
            warmup_iterations=100, timing_iterations=20, arch="sm_80",
        )

        assert b1.timing.raw_times_ms == b2.timing.raw_times_ms

    @pytest.mark.asyncio
    async def test_different_variants_different_results(self):
        """Different variant_ids produce different timing data."""
        runner = MockBenchmarkRunner(seed=42)
        b1 = await runner.benchmark(
            "/tmp/test", variant_id="v1", timing_iterations=10, arch="sm_80",
        )
        b2 = await runner.benchmark(
            "/tmp/test", variant_id="v2", timing_iterations=10, arch="sm_80",
        )
        assert b1.timing.raw_times_ms != b2.timing.raw_times_ms


# ── Benchmark Loop Exercised (via MinimalBenchmarkRunner) ──────

class TestBenchmarkLoopInherited:

    @pytest.mark.asyncio
    async def test_warmup_and_timing_iterations_exercised(self):
        """MinimalBenchmarkRunner.benchmark() calls run_kernel for warmup + timing."""
        runner = MinimalBenchmarkRunner()
        await runner.benchmark(
            "/tmp/test",
            variant_id="v1",
            warmup_iterations=3,
            timing_iterations=10,
            arch="sm_80",
        )
        # 3 warmup + 10 timing = 13 total run_kernel calls
        assert runner.call_count == 13

    @pytest.mark.asyncio
    async def test_different_warmup_counts_change_call_count(self):
        runner = MinimalBenchmarkRunner()
        await runner.benchmark(
            "/tmp/test", variant_id="v1",
            warmup_iterations=5, timing_iterations=10, arch="sm_80",
        )
        assert runner.call_count == 15  # 5 + 10

    @pytest.mark.asyncio
    async def test_timing_iterations_match_n_samples(self):
        runner = MinimalBenchmarkRunner()
        result = await runner.benchmark(
            "/tmp/test", variant_id="v1", timing_iterations=15, arch="sm_80"
        )
        assert result.timing.n_samples == 15


# ── Minimum Samples Enforcement (Critical Fix #1) ─────────────

class TestMinimumTimingIterations:

    @pytest.mark.asyncio
    async def test_rejects_timing_iterations_below_5(self):
        runner = MockBenchmarkRunner()
        with pytest.raises(BenchmarkError, match="timing_iterations must be >= 5"):
            await runner.benchmark(
                "/tmp/test", variant_id="v1", timing_iterations=1
            )

    @pytest.mark.asyncio
    async def test_rejects_timing_iterations_of_4(self):
        runner = MockBenchmarkRunner()
        with pytest.raises(BenchmarkError, match="timing_iterations must be >= 5"):
            await runner.benchmark(
                "/tmp/test", variant_id="v1", timing_iterations=4
            )

    @pytest.mark.asyncio
    async def test_accepts_timing_iterations_of_5(self):
        runner = MockBenchmarkRunner()
        result = await runner.benchmark(
            "/tmp/test", variant_id="v1", timing_iterations=5, arch="sm_80"
        )
        assert result.timing.n_samples == 5

    @pytest.mark.asyncio
    async def test_error_message_includes_value(self):
        runner = MockBenchmarkRunner()
        with pytest.raises(BenchmarkError, match="got 3"):
            await runner.benchmark(
                "/tmp/test", variant_id="v1", timing_iterations=3
            )

    @pytest.mark.asyncio
    async def test_warmup_not_run_when_validation_fails(self):
        """BenchmarkError raised before any run_kernel calls."""
        runner = MinimalBenchmarkRunner()
        initial = runner.call_count
        with pytest.raises(BenchmarkError):
            await runner.benchmark(
                "/tmp/test", variant_id="v1", timing_iterations=2
            )
        assert runner.call_count == initial


# ── Partial Failure Rejection (Critical Fix #1) ───────────────

class TestPartialFailureRejection:

    @pytest.mark.asyncio
    async def test_empty_run_kernel_raises_benchmark_error(self):
        """If run_kernel returns empty list, benchmark must raise."""
        runner = FailingRunKernelRunner()
        FailingRunKernelRunner.call_count = 0

        with pytest.raises(BenchmarkError, match="returned no data"):
            await runner.benchmark(
                "/tmp/test", variant_id="v1", timing_iterations=5, arch="sm_80",
            )

    @pytest.mark.asyncio
    async def test_error_identifies_iteration_number(self):
        """Error message tells which iteration failed."""
        runner = FailingRunKernelRunner()
        FailingRunKernelRunner.call_count = 0

        with pytest.raises(BenchmarkError, match="iteration 1/5"):
            await runner.benchmark(
                "/tmp/test", variant_id="v1", timing_iterations=5, arch="sm_80",
            )

    @pytest.mark.asyncio
    async def test_warmup_passes_then_timing_fails(self):
        """Warmup succeeds but timing fails — must still raise."""
        call_count = 0

        class WarmupOKTimingFailsRunner(BenchmarkRunner):
            MEASUREMENT_STRATEGY = MeasurementStrategy.MOCK

            async def run_kernel(self, binary_path, kernel_name=None, timeout_s=120.0):
                nonlocal call_count
                call_count += 1
                if call_count <= 3:
                    return [1.0]  # Warmup succeeds
                return []  # Timing fails

            async def is_available(self):
                return True

        runner = WarmupOKTimingFailsRunner()
        with pytest.raises(BenchmarkError, match="returned no data"):
            await runner.benchmark(
                "/tmp/test", variant_id="v1",
                warmup_iterations=3, timing_iterations=5, arch="sm_80",
            )


# ── Measurement Strategy (Critical Fix #2) ────────────────────

class TestMeasurementStrategy:

    @pytest.mark.asyncio
    async def test_mock_runner_strategy_is_mock(self):
        runner = MockBenchmarkRunner()
        result = await runner.benchmark(
            "/tmp/test", variant_id="v1", timing_iterations=10, arch="sm_80"
        )
        assert result.measurement_strategy == MeasurementStrategy.MOCK

    @pytest.mark.asyncio
    async def test_strategy_is_enum_instance(self):
        runner = MockBenchmarkRunner()
        result = await runner.benchmark(
            "/tmp/test", variant_id="v1", timing_iterations=10, arch="sm_80"
        )
        assert isinstance(result.measurement_strategy, MeasurementStrategy)

    @pytest.mark.asyncio
    async def test_strategy_serializes_as_string(self):
        """MeasurementStrategy inherits from str, so str() produces plain value."""
        runner = MockBenchmarkRunner()
        result = await runner.benchmark(
            "/tmp/test", variant_id="v1", timing_iterations=10, arch="sm_80"
        )
        assert str(result.measurement_strategy) == "mock"

    @pytest.mark.asyncio
    async def test_custom_subclass_strategy(self):
        class CudaEventRunner(MockBenchmarkRunner):
            MEASUREMENT_STRATEGY = MeasurementStrategy.CUDA_EVENT

        runner = CudaEventRunner()
        result = await runner.benchmark(
            "/tmp/test", variant_id="v1", timing_iterations=10, arch="sm_80"
        )
        assert result.measurement_strategy == MeasurementStrategy.CUDA_EVENT

    def test_measurement_strategy_enum_values(self):
        assert MeasurementStrategy.HOST == "host"
        assert MeasurementStrategy.CUDA_EVENT == "cuda_event"
        assert MeasurementStrategy.MOCK == "mock"

    def test_from_raw_string(self):
        assert MeasurementStrategy.from_raw("host") == MeasurementStrategy.HOST
        assert MeasurementStrategy.from_raw("cuda_event") == MeasurementStrategy.CUDA_EVENT
        assert MeasurementStrategy.from_raw("mock") == MeasurementStrategy.MOCK

    def test_from_raw_enum_passthrough(self):
        ms = MeasurementStrategy.HOST
        assert MeasurementStrategy.from_raw(ms) is ms

    def test_from_raw_invalid_raises(self):
        with pytest.raises(ValueError):
            MeasurementStrategy.from_raw("invalid_strategy")


# ── Stability Validation (Critical Fix #5) ────────────────────

class TestStabilityValidation:

    @pytest.mark.asyncio
    async def test_stable_benchmark_passes(self):
        """Low-variance preset with reasonable threshold passes."""
        runner = MockBenchmarkRunner()
        # tiled_matmul has std_ratio=0.02 (very stable)
        result = await runner.benchmark(
            "/tmp/tiled_matmul", variant_id="v1",
            timing_iterations=20, arch="sm_80",
            stability_cv_threshold=0.1,
        )
        assert result.timing.n_samples == 20

    @pytest.mark.asyncio
    async def test_unstable_benchmark_raises(self):
        """High CV threshold triggers stability error."""
        runner = MockBenchmarkRunner()
        # vecadd has std_ratio=0.05, but we set a very tight threshold
        with pytest.raises(BenchmarkError, match="unstable"):
            await runner.benchmark(
                "/tmp/vecadd", variant_id="v1",
                timing_iterations=20, arch="sm_80",
                stability_cv_threshold=0.001,
            )

    @pytest.mark.asyncio
    async def test_no_stability_check_by_default(self):
        """Without stability_cv_threshold, any variance is accepted."""
        runner = MockBenchmarkRunner()
        result = await runner.benchmark(
            "/tmp/test", variant_id="v1", timing_iterations=10, arch="sm_80",
        )
        # Should succeed without raising
        assert result.timing.n_samples == 10

    @pytest.mark.asyncio
    async def test_stability_error_includes_cv_and_threshold(self):
        runner = MockBenchmarkRunner()
        with pytest.raises(BenchmarkError, match="exceeds threshold"):
            await runner.benchmark(
                "/tmp/test", variant_id="v1",
                timing_iterations=20, arch="sm_80",
                stability_cv_threshold=0.0001,
            )

    @pytest.mark.asyncio
    async def test_stability_check_via_minimal_runner(self):
        """Stability validation also works through the base class benchmark loop."""
        runner = MinimalBenchmarkRunner()
        # Tight threshold should trigger
        with pytest.raises(BenchmarkError, match="unstable"):
            await runner.benchmark(
                "/tmp/test", variant_id="v1",
                timing_iterations=20, arch="sm_80",
                stability_cv_threshold=0.0001,
            )
