"""Tests for benchmark statistics computation."""

import math

import pytest

from cuda_opt_agent.benchmark.stats import (
    compute_timing_stats,
    coefficient_of_variation,
    is_stable,
    remove_outliers,
)
from cuda_opt_agent.schemas.benchmark import TimingStats


class TestComputeTimingStats:

    def test_basic_stats(self):
        times = [1.0, 2.0, 3.0, 4.0, 5.0]
        stats = compute_timing_stats(times, remove_outliers=False)

        assert stats.mean_ms == 3.0
        assert stats.median_ms == 3.0
        assert stats.min_ms == 1.0
        assert stats.max_ms == 5.0
        assert stats.n_samples == 5
        assert stats.std_ms > 0

    def test_single_sample(self):
        stats = compute_timing_stats([5.0], remove_outliers=False)
        assert stats.mean_ms == 5.0
        assert stats.median_ms == 5.0
        assert stats.std_ms == 0.0
        assert stats.n_samples == 1

    def test_empty_returns_zeros(self):
        stats = compute_timing_stats([])
        assert stats.mean_ms == 0.0
        assert stats.n_samples == 0

    def test_two_samples(self):
        stats = compute_timing_stats([1.0, 3.0], remove_outliers=False)
        assert stats.mean_ms == 2.0
        assert stats.median_ms == 2.0
        assert stats.n_samples == 2

    def test_percentiles_computed(self):
        times = [float(i) for i in range(1, 21)]
        stats = compute_timing_stats(times, remove_outliers=False)
        assert stats.p95_ms is not None
        assert stats.p99_ms is not None
        assert stats.p95_ms > stats.median_ms

    def test_percentiles_not_computed_few_samples(self):
        stats = compute_timing_stats([1.0, 2.0, 3.0], remove_outliers=False)
        assert stats.p95_ms is None
        assert stats.p99_ms is None

    def test_raw_times_preserved(self):
        times = [1.1, 2.2, 3.3]
        stats = compute_timing_stats(times, remove_outliers=False)
        assert len(stats.raw_times_ms) == 3

    def test_outlier_removal(self):
        times = [9.8, 9.9, 10.0, 10.0, 10.1, 10.0, 10.2, 9.9, 10.0, 100.0]
        stats = compute_timing_stats(times, remove_outliers=True)
        assert stats.mean_ms < 11.0
        assert stats.n_samples < 10

    def test_no_outlier_removal_when_disabled(self):
        times = [9.8, 9.9, 10.0, 10.0, 10.1, 10.0, 10.2, 9.9, 10.0, 100.0]
        stats = compute_timing_stats(times, remove_outliers=False)
        assert stats.n_samples == 10
        assert stats.mean_ms > 10.0


class TestFilteredTimes:
    """Tests for the filtered_times_ms field."""

    def test_none_when_no_outliers_removed(self):
        times = [10.0, 10.1, 10.2, 9.9, 10.0, 10.1]
        stats = compute_timing_stats(times, remove_outliers=True)
        assert stats.filtered_times_ms is None
        assert len(stats.raw_times_ms) == len(times)

    def test_populated_when_outliers_removed(self):
        times = [9.8, 9.9, 10.0, 10.0, 10.1, 10.0, 10.2, 9.9, 10.0, 100.0]
        stats = compute_timing_stats(times, remove_outliers=True)
        assert stats.filtered_times_ms is not None
        assert len(stats.filtered_times_ms) < len(times)
        assert 100.0 not in stats.filtered_times_ms
        assert len(stats.raw_times_ms) == len(times)
        assert 100.0 in stats.raw_times_ms

    def test_none_when_removal_disabled(self):
        times = [9.8, 9.9, 10.0, 10.0, 10.1, 10.0, 10.2, 9.9, 10.0, 100.0]
        stats = compute_timing_stats(times, remove_outliers=False)
        assert stats.filtered_times_ms is None

    def test_none_when_too_few_samples(self):
        times = [1.0, 2.0, 100.0]
        stats = compute_timing_stats(times, remove_outliers=True)
        assert stats.filtered_times_ms is None

    def test_stats_computed_on_filtered_data(self):
        times = [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 100.0]
        stats = compute_timing_stats(times, remove_outliers=True)
        assert stats.mean_ms < 11.0
        assert max(stats.raw_times_ms) == 100.0


class TestRemoveOutliers:

    def test_no_outliers(self):
        times = [10.0, 10.1, 10.2, 9.9, 10.0]
        result = remove_outliers(times)
        assert len(result) == 5

    def test_removes_extreme_outlier(self):
        times = [10.0, 10.1, 10.2, 9.9, 10.0, 100.0]
        result = remove_outliers(times)
        assert 100.0 not in result

    def test_too_few_samples(self):
        times = [1.0, 2.0, 3.0]
        result = remove_outliers(times)
        assert len(result) == 3

    def test_empty(self):
        assert remove_outliers([]) == []


class TestStability:

    def test_stable_measurement(self):
        stats = TimingStats(
            mean_ms=10.0, median_ms=10.0, std_ms=0.5,
            min_ms=9.0, max_ms=11.0, n_samples=10,
        )
        assert coefficient_of_variation(stats) == 0.05
        assert is_stable(stats, cv_threshold=0.1) is True

    def test_unstable_measurement(self):
        stats = TimingStats(
            mean_ms=10.0, median_ms=10.0, std_ms=3.0,
            min_ms=5.0, max_ms=15.0, n_samples=10,
        )
        assert coefficient_of_variation(stats) == 0.3
        assert is_stable(stats, cv_threshold=0.1) is False

    def test_zero_mean(self):
        stats = TimingStats(
            mean_ms=0.0, median_ms=0.0, std_ms=0.0,
            min_ms=0.0, max_ms=0.0, n_samples=1,
        )
        assert coefficient_of_variation(stats) == float("inf")
