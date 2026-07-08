"""Abstract search strategy and supporting types.

A SearchStrategy generates candidate TransformSpec objects for the
coordinator to evaluate. It supports iterative evaluation:
1. coordinator calls generate_candidates()
2. coordinator evaluates candidates (compile, benchmark)
3. coordinator calls report_result() for each outcome
4. coordinator calls generate_candidates() again until done

Grid search pre-generates all candidates and ignores report_result().
Adaptive strategies (Bayesian, evolutionary) use report_result() to
improve subsequent candidate generation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from cuda_opt_agent.schemas.transform import TransformSpec
from cuda_opt_agent.transforms.base import Transformation


@dataclass
class SearchResult:
    """Outcome of evaluating a single candidate TransformSpec."""

    spec: TransformSpec
    speedup: float = 1.0
    correct: bool = True
    benchmark_time_ms: float | None = None
    error: str | None = None

    @property
    def is_improvement(self) -> bool:
        return self.speedup > 1.0 and self.correct


@dataclass
class SearchContext:
    """Information available to the search strategy for candidate generation."""

    transforms: list[Transformation]
    iteration: int = 0
    best_speedup: float = 1.0
    previous_results: list[SearchResult] = field(default_factory=list)

    @property
    def evaluated_count(self) -> int:
        return len(self.previous_results)

    @property
    def improvement_count(self) -> int:
        return sum(1 for r in self.previous_results if r.is_improvement)


class SearchStrategy(ABC):
    """Generate candidate TransformSpec objects for evaluation.

    Lifecycle:
    - generate_candidates() called repeatedly until it returns []
    - report_result() called after each evaluation (optional for non-adaptive)
    - reset() called to restart the search
    """

    @abstractmethod
    def generate_candidates(
        self, context: SearchContext
    ) -> list[TransformSpec]:
        """Generate the next batch of candidates.

        Returns an empty list when the search is exhausted.
        For non-adaptive strategies (grid), returns all remaining candidates.
        For adaptive strategies (bayesian), returns a small batch and adapts
        based on report_result() feedback.
        """

    def report_result(self, result: SearchResult) -> None:
        """Report evaluation outcome. Default: no-op.

        Adaptive strategies override this to learn from results.
        """

    def reset(self) -> None:
        """Reset internal state for a fresh search. Default: no-op."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Strategy identifier (e.g., 'grid', 'random')."""

    @property
    def is_adaptive(self) -> bool:
        """True for strategies that use report_result() to adapt."""
        return False


# ── Parameter schema parsing ───────────────────────────────────

def extract_param_grid(schema: dict[str, Any]) -> dict[str, list[Any]]:
    """Parse a Transform's parameter_schema into a grid of values per parameter.

    For each property in the JSON Schema:
    - "enum" values: use all
    - "integer"/"number" with min+max: generate representative values
    - "default" only: use just the default
    - No info: skip

    Returns:
        Dict mapping parameter name → list of candidate values.
    """
    properties = schema.get("properties", {})
    grid: dict[str, list[Any]] = {}

    for param_name, param_spec in properties.items():
        values = _extract_values(param_spec)
        if values:
            grid[param_name] = values

    return grid


def _extract_values(spec: dict[str, Any]) -> list[Any]:
    """Extract candidate values for a single parameter from its JSON Schema spec."""
    # Enum takes priority
    if "enum" in spec:
        return list(spec["enum"])

    param_type = spec.get("type", "")
    default = spec.get("default")
    minimum = spec.get("minimum")
    maximum = spec.get("maximum")

    if param_type == "integer" and minimum is not None and maximum is not None:
        return _int_grid(int(minimum), int(maximum))

    if param_type == "number" and minimum is not None and maximum is not None:
        return _float_grid(float(minimum), float(maximum), steps=5)

    if default is not None:
        return [default]

    return []


def _int_grid(lo: int, hi: int) -> list[int]:
    """Generate a representative integer grid from lo to hi (inclusive).

    Strategy: powers of 2 in range, plus lo and hi.
    Limited to at most 8 values to keep grid size manageable.
    """
    values: set[int] = {lo, hi}
    power = 1
    while power <= hi:
        if lo <= power <= hi:
            values.add(power)
        power *= 2

    # Fill with evenly-spaced values if too few
    if len(values) < 4 and hi - lo >= 3:
        step = max(1, (hi - lo) // 4)
        for v in range(lo, hi + 1, step):
            values.add(v)

    result = sorted(values)
    return result[:8]  # Cap at 8 to prevent combinatorial explosion


def _float_grid(lo: float, hi: float, steps: int = 5) -> list[float]:
    """Generate evenly-spaced float values from lo to hi."""
    if steps <= 1:
        return [lo]
    step = (hi - lo) / (steps - 1)
    return [round(lo + i * step, 6) for i in range(steps)]
