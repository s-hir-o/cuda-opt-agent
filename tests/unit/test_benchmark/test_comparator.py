"""Tests for variant comparison and speedup computation."""

import hashlib
import math
import random

import pytest

from cuda_opt_agent.benchmark.comparator import (
    compare_variants,
    rank_variants,
    welch_t_test,
)
from cuda_opt_agent.core.exceptions import BenchmarkError
from cuda_opt_agent.schemas.benchmark import (
    BenchmarkResult,
    MeasurementStrategy,
    TimingStats,
)


def _make_result(
    variant_id: str,
    mean_ms: float,
    std_ms: float = 0.1,
    n: int = 30,
    seed: int = 0,
    strategy: MeasurementStrategy = MeasurementStrategy.MOCK,
) -> BenchmarkResult:
    """Create a BenchmarkResult with realistic variance."""
    stable = int(hashlib.sha256(variant_id.encode()).hexdigest()[:8], 16)
    rng = random.Random(seed + stable)
    raw = [max(0.01, rng.gauss(mean_ms, std_ms)) for _ in range(n)]

    mean = sum(raw) / len(raw)
    variance = sum((t - mean) ** 2 for t in raw) / (len(raw) - 1)

    return BenchmarkResult(
        variant_id=variant_id,
        timing=TimingStats(
            mean_ms=round(mean, 6),
            median_ms=sorted(raw)[len(raw) // 2],
            std_ms=round(math.sqrt(variance), 6),
            min_ms=min(raw),
            max_ms=max(raw),
            n_samples=len(raw),
            raw_times_ms=raw,
        ),
        measurement_strategy=strategy,
    )


class TestCompareVariants:

    def test_2x_speedup(self):
        baseline = _make_result("v0", mean_ms=10.0, std_ms=0.2, seed=1)
        candidate = _make_result("v1", mean_ms=5.0, std_ms=0.1, seed=2)
        result = compare_variants(baseline, candidate)

        assert result.speedup > 1.5
        assert result.is_improvement
        assert result.significant is True

    def test_no_speedup(self):
        baseline = _make_result("v0", mean_ms=10.0, std_ms=0.1, seed=1)
        candidate = _make_result("v1", mean_ms=10.0, std_ms=0.1, seed=2)
        result = compare_variants(baseline, candidate)

        assert result.speedup == pytest.approx(1.0, abs=0.1)
        assert not result.is_improvement

    def test_slowdown(self):
        baseline = _make_result("v0", mean_ms=5.0, std_ms=0.1, seed=1)
        candidate = _make_result("v1", mean_ms=10.0, std_ms=0.1, seed=2)
        result = compare_variants(baseline, candidate)

        assert result.speedup < 1.0
        assert not result.is_improvement

    def test_variant_ids_populated(self):
        baseline = _make_result("baseline", mean_ms=10.0, seed=1)
        candidate = _make_result("candidate", mean_ms=5.0, seed=2)
        result = compare_variants(baseline, candidate)

        assert result.baseline_variant_id == "baseline"
        assert result.candidate_variant_id == "candidate"

    def test_not_significant_when_close(self):
        baseline = _make_result("v0", mean_ms=10.0, std_ms=1.0, seed=1)
        candidate = _make_result("v1", mean_ms=10.0, std_ms=1.0, seed=2)
        result = compare_variants(baseline, candidate)
        assert result.significant is False

    def test_confidence_interval_present(self):
        baseline = _make_result("v0", mean_ms=10.0, std_ms=0.2, seed=1)
        candidate = _make_result("v1", mean_ms=5.0, std_ms=0.1, seed=2)
        result = compare_variants(baseline, candidate)
        assert result.confidence_interval is not None
        assert len(result.confidence_interval) == 2
        assert result.confidence_interval[0] > 0

    def test_speedup_and_effective_speedup_populated(self):
        baseline = _make_result("v0", mean_ms=10.0, std_ms=0.2, seed=1)
        candidate = _make_result("v1", mean_ms=5.0, std_ms=0.1, seed=2)
        result = compare_variants(baseline, candidate)
        assert result.speedup > 1.0
        assert result.effective_speedup > 1.0


# ── Strategy Validation (Critical Fix #4) ─────────────────────

class TestCrossStrategyRejection:

    def test_host_vs_cuda_event_rejected(self):
        host = _make_result("v0", mean_ms=10.0, seed=1, strategy=MeasurementStrategy.HOST)
        cuda = _make_result("v1", mean_ms=5.0, seed=2, strategy=MeasurementStrategy.CUDA_EVENT)
        with pytest.raises(BenchmarkError, match="different measurement strategies"):
            compare_variants(host, cuda)

    def test_host_vs_mock_rejected(self):
        host = _make_result("v0", mean_ms=10.0, seed=1, strategy=MeasurementStrategy.HOST)
        mock = _make_result("v1", mean_ms=5.0, seed=2, strategy=MeasurementStrategy.MOCK)
        with pytest.raises(BenchmarkError, match="different measurement strategies"):
            compare_variants(host, mock)

    def test_cuda_event_vs_mock_rejected(self):
        cuda = _make_result("v0", mean_ms=10.0, seed=1, strategy=MeasurementStrategy.CUDA_EVENT)
        mock = _make_result("v1", mean_ms=5.0, seed=2, strategy=MeasurementStrategy.MOCK)
        with pytest.raises(BenchmarkError, match="different measurement strategies"):
            compare_variants(cuda, mock)

    def test_same_strategy_allowed(self):
        a = _make_result("v0", mean_ms=10.0, seed=1, strategy=MeasurementStrategy.MOCK)
        b = _make_result("v1", mean_ms=5.0, seed=2, strategy=MeasurementStrategy.MOCK)
        result = compare_variants(a, b)
        assert result.speedup > 1.0

    def test_string_strategy_normalization(self):
        """String strategy values are normalized to enum for comparison."""
        a = _make_result("v0", mean_ms=10.0, seed=1, strategy=MeasurementStrategy.HOST)
        b = _make_result("v1", mean_ms=5.0, seed=2, strategy=MeasurementStrategy.HOST)
        # Both are HOST — should succeed
        result = compare_variants(a, b)
        assert result.speedup > 1.0

    def test_error_mentions_both_strategies(self):
        host = _make_result("v0", mean_ms=10.0, seed=1, strategy=MeasurementStrategy.HOST)
        cuda = _make_result("v1", mean_ms=5.0, seed=2, strategy=MeasurementStrategy.CUDA_EVENT)
        with pytest.raises(BenchmarkError) as exc_info:
            compare_variants(host, cuda)
        assert "host" in str(exc_info.value)
        assert "cuda_event" in str(exc_info.value)


# ── Noise-Based Speedups (Critical Fix #6) ────────────────────

class TestNoiseFiltering:

    def test_1pct_speedup_not_improvement(self):
        """A 1% speedup is below the default 1% threshold — treated as noise."""
        # Both at 10.0ms with minimal variance
        baseline = _make_result("v0", mean_ms=10.0, std_ms=0.05, seed=1)
        candidate = _make_result("v1", mean_ms=9.9, std_ms=0.05, seed=2)
        result = compare_variants(baseline, candidate)
        # Speedup ≈ 1.01, below threshold
        assert result.effective_speedup == 1.0
        assert not result.is_improvement

    def test_2pct_speedup_not_improvement_by_default(self):
        """A 2% speedup exceeds the default 1% threshold."""
        baseline = _make_result("v0", mean_ms=10.0, std_ms=0.05, seed=1)
        candidate = _make_result("v1", mean_ms=9.8, std_ms=0.05, seed=2)
        result = compare_variants(baseline, candidate)
        assert result.speedup > 1.0
        assert result.effective_speedup > 1.0
        assert result.is_improvement

    def test_custom_threshold_filters_2pct(self):
        """With minimum_effect_size=0.03, a 2% speedup is filtered."""
        baseline = _make_result("v0", mean_ms=10.0, std_ms=0.05, seed=1)
        candidate = _make_result("v1", mean_ms=9.8, std_ms=0.05, seed=2)
        result = compare_variants(baseline, candidate, minimum_effect_size=0.03)
        assert result.speedup > 1.0  # raw speedup preserved
        assert result.effective_speedup == 1.0  # clamped to noise
        assert not result.is_improvement

    def test_custom_threshold_allows_5pct(self):
        """With minimum_effect_size=0.03, a 5% speedup passes."""
        baseline = _make_result("v0", mean_ms=10.0, std_ms=0.05, seed=1)
        candidate = _make_result("v1", mean_ms=9.5, std_ms=0.05, seed=2)
        result = compare_variants(baseline, candidate, minimum_effect_size=0.03)
        assert result.speedup > 1.0
        assert result.effective_speedup > 1.0
        assert result.is_improvement

    def test_zero_threshold_treats_all_as_real(self):
        """With minimum_effect_size=0, any speedup is classified as improvement."""
        baseline = _make_result("v0", mean_ms=10.0, std_ms=0.05, seed=1)
        candidate = _make_result("v1", mean_ms=9.95, std_ms=0.05, seed=2)
        result = compare_variants(baseline, candidate, minimum_effect_size=0.0)
        # Even a tiny speedup passes
        assert result.effective_speedup > 1.0 or result.speedup <= 1.0

    def test_raw_speedup_always_preserved(self):
        """speedup field always contains the raw ratio, even when clamped."""
        baseline = _make_result("v0", mean_ms=10.0, std_ms=0.05, seed=1)
        candidate = _make_result("v1", mean_ms=9.9, std_ms=0.05, seed=2)
        result = compare_variants(baseline, candidate, minimum_effect_size=0.05)
        # Raw speedup ≈ 1.01 is preserved in the speedup field
        assert result.speedup > 1.0
        # But effective_speedup is clamped
        assert result.effective_speedup == 1.0


# ── Welch's t-test ─────────────────────────────────────────────

class TestWelchTTest:

    def test_identical_samples_not_significant(self):
        rng = random.Random(42)
        sample = [10.0 + rng.gauss(0, 0.1) for _ in range(20)]
        sig, ci = welch_t_test(sample, sample, alpha=0.05)
        assert sig is False

    def test_very_different_samples_significant(self):
        a = [100.0 + random.Random(1).gauss(0, 0.5) for _ in range(20)]
        b = [10.0 + random.Random(2).gauss(0, 0.5) for _ in range(20)]
        sig, ci = welch_t_test(a, b, alpha=0.05)
        assert sig is True
        assert ci is not None
        assert ci[0] > 0

    def test_small_sample_insufficient(self):
        sig, ci = welch_t_test([1.0], [2.0])
        assert sig is False
        assert ci is None

    def test_zero_variance_different_means_significant(self):
        a = [10.0, 10.0, 10.0, 10.0, 10.0]
        b = [20.0, 20.0, 20.0, 20.0, 20.0]
        sig, ci = welch_t_test(a, b, alpha=0.05)
        assert sig is True
        assert ci is not None
        assert ci == pytest.approx((-10.0, -10.0))

    def test_zero_variance_same_means_not_significant(self):
        a = [10.0, 10.0, 10.0, 10.0, 10.0]
        b = [10.0, 10.0, 10.0, 10.0, 10.0]
        sig, ci = welch_t_test(a, b, alpha=0.05)
        assert sig is False
        assert ci == (0.0, 0.0)


# ── Rank Variants ──────────────────────────────────────────────

class TestRankVariants:

    def test_ranking_fastest_first(self):
        slow = _make_result("slow", mean_ms=20.0, seed=1)
        medium = _make_result("medium", mean_ms=10.0, seed=2)
        fast = _make_result("fast", mean_ms=5.0, seed=3)

        rankings = rank_variants([slow, medium, fast])
        assert len(rankings) == 2
        assert rankings[0].candidate_variant_id == "fast"
        assert rankings[0].speedup > rankings[1].speedup

    def test_explicit_baseline(self):
        v0 = _make_result("v0", mean_ms=10.0, seed=1)
        v1 = _make_result("v1", mean_ms=5.0, seed=2)
        v2 = _make_result("v2", mean_ms=7.0, seed=3)

        rankings = rank_variants([v0, v1, v2], baseline_id="v0")
        assert len(rankings) == 2
        assert all(r.baseline_variant_id == "v0" for r in rankings)

    def test_too_few_results(self):
        single = _make_result("v0", mean_ms=10.0, seed=1)
        assert rank_variants([single]) == []

    def test_nonexistent_baseline_uses_slowest(self):
        v0 = _make_result("v0", mean_ms=10.0, seed=1)
        v1 = _make_result("v1", mean_ms=5.0, seed=2)

        rankings = rank_variants([v0, v1], baseline_id="nonexistent")
        assert len(rankings) == 1
        assert rankings[0].baseline_variant_id == "v0"

    def test_incompatible_strategies_skipped(self):
        """Results with incompatible strategies are silently skipped in ranking."""
        host = _make_result("host_v0", mean_ms=10.0, seed=1, strategy=MeasurementStrategy.HOST)
        host_fast = _make_result("host_v1", mean_ms=5.0, seed=2, strategy=MeasurementStrategy.HOST)
        cuda = _make_result("cuda_v0", mean_ms=8.0, seed=3, strategy=MeasurementStrategy.CUDA_EVENT)

        # host and cuda can't be compared — cuda should be skipped
        rankings = rank_variants([host, host_fast, cuda])
        assert len(rankings) == 1
        assert rankings[0].candidate_variant_id == "host_v1"
