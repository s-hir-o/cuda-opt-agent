"""Statistical analysis of benchmark timing data.

Computes TimingStats from raw millisecond measurements.
Includes outlier removal via IQR method and percentile computation.
"""

from __future__ import annotations

import math

from cuda_opt_agent.schemas.benchmark import TimingStats


def compute_timing_stats(
    raw_times_ms: list[float],
    remove_outliers: bool = True,
) -> TimingStats:
    """Compute TimingStats from raw millisecond measurements.

    Args:
        raw_times_ms: List of individual timing measurements in ms.
        remove_outliers: If True, removes IQR outliers before computing stats.

    Returns:
        TimingStats with mean, median, std, min, max, percentiles.
    """
    if not raw_times_ms:
        return TimingStats(
            mean_ms=0.0, median_ms=0.0, std_ms=0.0,
            min_ms=0.0, max_ms=0.0, n_samples=0,
        )

    times = list(raw_times_ms)
    filtered_times: list[float] | None = None
    if remove_outliers and len(times) >= 5:
        filtered = _remove_iqr_outliers(times)
        if len(filtered) != len(times):
            filtered_times = [round(t, 6) for t in filtered]
            times = filtered

    if not times:
        times = list(raw_times_ms)
        filtered_times = None

    times_sorted = sorted(times)
    n = len(times_sorted)

    mean = sum(times_sorted) / n
    median = _percentile(times_sorted, 50.0)
    variance = sum((t - mean) ** 2 for t in times_sorted) / max(n - 1, 1)
    std = math.sqrt(variance)

    p95 = _percentile(times_sorted, 95.0) if n >= 5 else None
    p99 = _percentile(times_sorted, 99.0) if n >= 10 else None

    return TimingStats(
        mean_ms=round(mean, 6),
        median_ms=round(median, 6),
        std_ms=round(std, 6),
        min_ms=round(times_sorted[0], 6),
        max_ms=round(times_sorted[-1], 6),
        p95_ms=round(p95, 6) if p95 is not None else None,
        p99_ms=round(p99, 6) if p99 is not None else None,
        n_samples=n,
        raw_times_ms=[round(t, 6) for t in raw_times_ms],
        filtered_times_ms=filtered_times,
    )


def remove_outliers(times_ms: list[float]) -> list[float]:
    """Remove IQR outliers from timing data. Public wrapper."""
    if len(times_ms) < 5:
        return list(times_ms)
    return _remove_iqr_outliers(times_ms)


def _remove_iqr_outliers(times: list[float], k: float = 1.5) -> list[float]:
    """Remove outliers using the IQR method.

    Points outside [Q1 - k*IQR, Q3 + k*IQR] are removed.
    Standard k=1.5 for "mild" outliers.
    """
    sorted_t = sorted(times)
    n = len(sorted_t)
    q1 = _percentile(sorted_t, 25.0)
    q3 = _percentile(sorted_t, 75.0)
    iqr = q3 - q1

    if iqr == 0:
        median = _percentile(sorted_t, 50.0)
        filtered = [t for t in times if t == median]
        return filtered if filtered else list(times)

    lower = q1 - k * iqr
    upper = q3 + k * iqr

    return [t for t in times if lower <= t <= upper]


def _percentile(sorted_data: list[float], p: float) -> float:
    """Compute the p-th percentile from pre-sorted data using linear interpolation.

    Args:
        sorted_data: Pre-sorted list of values.
        p: Percentile (0-100).

    Returns:
        Interpolated percentile value.
    """
    n = len(sorted_data)
    if n == 0:
        return 0.0
    if n == 1:
        return sorted_data[0]

    # Use linear interpolation method (same as numpy.percentile default)
    rank = (p / 100.0) * (n - 1)
    lower_idx = int(math.floor(rank))
    upper_idx = min(lower_idx + 1, n - 1)
    frac = rank - lower_idx

    return sorted_data[lower_idx] * (1 - frac) + sorted_data[upper_idx] * frac


def coefficient_of_variation(stats: TimingStats) -> float:
    """Compute CV (std/mean). Lower = more stable measurement."""
    if stats.mean_ms <= 0:
        return float("inf")
    return stats.std_ms / stats.mean_ms


def is_stable(stats: TimingStats, cv_threshold: float = 0.1) -> bool:
    """Return True if measurement is stable (CV below threshold)."""
    return coefficient_of_variation(stats) < cv_threshold
