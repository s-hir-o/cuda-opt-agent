"""Variant comparison — speedup computation with statistical significance.

Compares a candidate variant against a baseline using Welch's t-test
for unequal variances to determine if the speedup is statistically
significant.

Validates that both results use the same measurement_strategy to prevent
meaningless cross-strategy comparisons.
"""

from __future__ import annotations

import math

from cuda_opt_agent.core.exceptions import BenchmarkError
from cuda_opt_agent.schemas.benchmark import (
    BenchmarkResult,
    MeasurementStrategy,
    SpeedupResult,
    TimingStats,
)

# Default minimum speedup ratio to classify as improvement.
# A speedup of 1.02 means the candidate is 2% faster — below this
# threshold, differences are treated as measurement noise.
DEFAULT_MIN_EFFECT_SIZE = 0.01


def _resolve_strategy(
    result: BenchmarkResult, context: str = "result"
) -> MeasurementStrategy:
    """Extract and normalize the measurement strategy from a BenchmarkResult."""
    s = result.measurement_strategy
    if not s:
        return MeasurementStrategy.HOST
    if isinstance(s, MeasurementStrategy):
        return s
    return MeasurementStrategy.from_raw(s)


def compare_variants(
    baseline: BenchmarkResult,
    candidate: BenchmarkResult,
    significance_level: float = 0.05,
    minimum_effect_size: float = DEFAULT_MIN_EFFECT_SIZE,
) -> SpeedupResult:
    """Compare candidate variant against baseline.

    Args:
        baseline: BenchmarkResult for the baseline kernel.
        candidate: BenchmarkResult for the candidate kernel.
        significance_level: p-value threshold for significance (default 0.05).
        minimum_effect_size: Speedups below this ratio above 1.0 are
            treated as noise and will not be classified as improvements.
            Default 0.01 (1%).

    Returns:
        SpeedupResult with speedup factor and significance flag.

    Raises:
        BenchmarkError: If the two results used different measurement
            strategies (e.g., host vs cuda_event).
    """
    b_strategy = _resolve_strategy(baseline, "baseline")
    c_strategy = _resolve_strategy(candidate, "candidate")

    if b_strategy != c_strategy:
        raise BenchmarkError(
            f"Cannot compare results with different measurement strategies: "
            f"{b_strategy.value} vs {c_strategy.value}"
        )

    b_stats = baseline.timing
    c_stats = candidate.timing

    baseline_time = b_stats.median_ms
    candidate_time = c_stats.median_ms

    if candidate_time <= 0:
        speedup = float("inf") if baseline_time > 0 else 1.0
    else:
        speedup = baseline_time / candidate_time

    # Apply minimum effect size — clamp tiny differences to 1.0
    threshold_tolerance = 0.002 if minimum_effect_size > 0 else 0.0
    if abs(speedup - 1.0) <= minimum_effect_size + threshold_tolerance:
        effective_speedup = 1.0
    else:
        effective_speedup = speedup

    # Statistical significance via Welch's t-test
    significant, ci = welch_t_test(
        b_stats.raw_times_ms or [b_stats.mean_ms],
        c_stats.raw_times_ms or [c_stats.mean_ms],
        significance_level,
    )

    return SpeedupResult(
        baseline_variant_id=baseline.variant_id,
        candidate_variant_id=candidate.variant_id,
        baseline_time_ms=round(baseline_time, 6),
        candidate_time_ms=round(candidate_time, 6),
        speedup=round(speedup, 4),
        significant=significant,
        confidence_interval=ci,
        effective_speedup=round(effective_speedup, 4),
    )


def welch_t_test(
    sample_a: list[float],
    sample_b: list[float],
    alpha: float = 0.05,
) -> tuple[bool, tuple[float, float] | None]:
    """Welch's t-test for unequal variances.

    Tests H0: mean(a) == mean(b) against H1: mean(a) != mean(b).

    Args:
        sample_a: Baseline timing samples.
        sample_b: Candidate timing samples.
        alpha: Significance level.

    Returns:
        Tuple of (reject_h0, confidence_interval_for_difference).
        reject_h0 is True if the difference is significant.
        Confidence interval is for (mean_a - mean_b).
    """
    n_a = len(sample_a)
    n_b = len(sample_b)

    if n_a < 2 or n_b < 2:
        return False, None

    mean_a = sum(sample_a) / n_a
    mean_b = sum(sample_b) / n_b

    var_a = sum((x - mean_a) ** 2 for x in sample_a) / (n_a - 1)
    var_b = sum((x - mean_b) ** 2 for x in sample_b) / (n_b - 1)

    se_a = var_a / n_a
    se_b = var_b / n_b
    se_diff = math.sqrt(se_a + se_b)

    diff = mean_a - mean_b

    if se_diff == 0:
        # Both samples have zero variance.
        if abs(diff) > 0:
            return True, (round(diff, 6), round(diff, 6))
        return False, (0.0, 0.0)

    # t-statistic
    t_stat = diff / se_diff

    # Welch-Satterthwaite degrees of freedom
    numerator = (se_a + se_b) ** 2
    denominator = (se_a ** 2) / (n_a - 1) + (se_b ** 2) / (n_b - 1)
    df = numerator / denominator if denominator > 0 else 1.0

    # p-value approximation
    p_value = _two_tailed_p_value(t_stat, df)

    # Confidence interval for difference (mean_a - mean_b)
    t_crit = _t_critical(alpha / 2, df)
    margin = t_crit * se_diff
    ci = (round(diff - margin, 6), round(diff + margin, 6))

    significant = p_value < alpha
    return significant, ci


def _two_tailed_p_value(t_stat: float, df: float) -> float:
    """Approximate two-tailed p-value using the normal distribution."""
    abs_t = abs(t_stat)

    if df > 100:
        return 2.0 * _normal_sf(abs_t)

    correction = 1.0 - 1.0 / (4.0 * df)
    z = abs_t * correction
    return 2.0 * _normal_sf(z)


def _normal_sf(x: float) -> float:
    """Survival function (1 - CDF) of standard normal distribution."""
    return 0.5 * math.erfc(x / math.sqrt(2))


def _t_critical(alpha: float, df: float) -> float:
    """Approximate critical value of t-distribution."""
    if df > 100:
        return _normal_quantile(1 - alpha)

    z = _normal_quantile(1 - alpha)
    return z + (z ** 3 + z) / (4.0 * df)


def _normal_quantile(p: float) -> float:
    """Approximate quantile (inverse CDF) of standard normal.

    Uses the rational approximation from Abramowitz & Stegun 26.2.23.
    """
    if p <= 0:
        return float("-inf")
    if p >= 1:
        return float("inf")
    if p == 0.5:
        return 0.0

    if p > 0.5:
        return -_normal_quantile(1 - p)

    # Rational approximation for 0 < p < 0.5
    t = math.sqrt(-2.0 * math.log(p))
    c0, c1, c2 = 2.515517, 0.802853, 0.010328
    d1, d2, d3 = 1.432788, 0.189269, 0.001308
    return -(t - (c0 + c1 * t + c2 * t ** 2) / (1 + d1 * t + d2 * t ** 2 + d3 * t ** 3))


def rank_variants(
    results: list[BenchmarkResult],
    baseline_id: str | None = None,
    minimum_effect_size: float = DEFAULT_MIN_EFFECT_SIZE,
) -> list[SpeedupResult]:
    """Rank a list of benchmark results by speedup against a baseline.

    If baseline_id is None, uses the slowest variant as baseline.

    Args:
        results: List of BenchmarkResult to rank.
        baseline_id: Variant ID of the baseline. If None, uses slowest.
        minimum_effect_size: Minimum speedup ratio to classify as improvement.

    Returns:
        List of SpeedupResult sorted by speedup (fastest first).
    """
    if len(results) < 2:
        return []

    if baseline_id:
        baseline = next((r for r in results if r.variant_id == baseline_id), None)
        if baseline is None:
            baseline = max(results, key=lambda r: r.timing.median_ms)
    else:
        baseline = max(results, key=lambda r: r.timing.median_ms)

    comparisons: list[SpeedupResult] = []
    for result in results:
        if result.variant_id == baseline.variant_id:
            continue
        try:
            comparisons.append(
                compare_variants(baseline, result, minimum_effect_size=minimum_effect_size)
            )
        except BenchmarkError:
            # Skip results with incompatible measurement strategies
            continue

    comparisons.sort(key=lambda s: s.speedup, reverse=True)
    return comparisons
