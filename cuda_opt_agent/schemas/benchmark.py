"""Benchmark schemas — data models for timing, results, and comparisons."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class MeasurementStrategy(str, Enum):
    """Valid benchmark timing strategies.

    Inherits from str so instances serialize to plain strings in JSON,
    database storage, and all existing string-based comparisons continue
    to work unchanged.
    """

    HOST = "host"
    CUDA_EVENT = "cuda_event"
    MOCK = "mock"

    def __str__(self) -> str:
        """Return the serialized strategy value."""
        return self.value

    @classmethod
    def from_raw(cls, value: "MeasurementStrategy | str") -> "MeasurementStrategy":
        """Accept a string or MeasurementStrategy, always return enum instance."""
        if isinstance(value, cls):
            return value
        return cls(value)


@dataclass
class TimingStats:
    """Statistical summary of timing measurements.

    raw_times_ms always contains the original unmodified measurements.
    filtered_times_ms is populated only when outlier removal actually
    changed the data (i.e., at least one point was removed).
    """

    mean_ms: float
    median_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    n_samples: int
    p95_ms: float | None = None
    p99_ms: float | None = None
    raw_times_ms: list[float] = field(default_factory=list)
    filtered_times_ms: list[float] | None = None

    def model_dump(self, mode: str = "python") -> dict[str, Any]:
        """Compatibility with the Pydantic schema API used by repositories."""
        return asdict(self)

    @classmethod
    def model_validate(cls, data: "TimingStats | dict[str, Any]") -> "TimingStats":
        """Compatibility with the Pydantic schema API used by repositories."""
        if isinstance(data, cls):
            return data
        return cls(**data)


@dataclass
class BenchmarkResult:
    """Result of benchmarking a kernel variant.

    measurement_strategy records how the timing was obtained:
    "host"      — host-side wall-clock (includes process/launch overhead)
    "cuda_event" — GPU-only via CUDA events (precise)
    "mock"      — simulated timing (no GPU)
    """

    variant_id: str
    timing: TimingStats
    arch: str = ""
    block_size: tuple[int, ...] | None = None
    grid_size: tuple[int, ...] | None = None
    correct: bool = True
    measurement_strategy: MeasurementStrategy | str = ""
    benchmarked_at: datetime = field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)

    def model_dump(self, mode: str = "python") -> dict[str, Any]:
        """Compatibility with the Pydantic schema API."""
        data = asdict(self)
        data["timing"] = self.timing.model_dump(mode=mode)
        if isinstance(self.measurement_strategy, MeasurementStrategy):
            data["measurement_strategy"] = self.measurement_strategy.value
        return data

    @classmethod
    def model_validate(
        cls, data: "BenchmarkResult | dict[str, Any]"
    ) -> "BenchmarkResult":
        """Compatibility with the Pydantic schema API."""
        if isinstance(data, cls):
            return data
        values = dict(data)
        values["timing"] = TimingStats.model_validate(values["timing"])
        if values.get("measurement_strategy"):
            values["measurement_strategy"] = MeasurementStrategy.from_raw(
                values["measurement_strategy"]
            )
        return cls(**values)


@dataclass
class SpeedupResult:
    """Comparison between two benchmark results.

    effective_speedup applies the minimum_effect_size threshold:
    if the raw speedup is within 1.0 ± threshold of 1.0, it is clamped
    to 1.0 so that is_improvement returns False.
    """

    baseline_variant_id: str
    candidate_variant_id: str
    baseline_time_ms: float
    candidate_time_ms: float
    speedup: float
    significant: bool
    confidence_interval: tuple[float, float] | None = None
    effective_speedup: float = 1.0

    @property
    def is_improvement(self) -> bool:
        """True if the candidate is meaningfully faster than the baseline."""
        return self.effective_speedup > 1.0
