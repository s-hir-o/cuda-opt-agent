"""Benchmark layer — kernel timing, statistics, and variant comparison."""

from cuda_opt_agent.benchmark.runner import BenchmarkRunner
from cuda_opt_agent.benchmark.timer import HostTimer, CudaEventTimer
from cuda_opt_agent.benchmark.stats import compute_timing_stats, remove_outliers
from cuda_opt_agent.benchmark.comparator import compare_variants
from cuda_opt_agent.benchmark.mock_runner import MockBenchmarkRunner

__all__ = [
    "BenchmarkRunner",
    "HostTimer",
    "CudaEventTimer",
    "compute_timing_stats",
    "remove_outliers",
    "compare_variants",
    "MockBenchmarkRunner",
]
