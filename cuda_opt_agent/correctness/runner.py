"""Abstract correctness comparison interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

from cuda_opt_agent.schemas.compilation import CompilationResult
from cuda_opt_agent.schemas.correctness import CorrectnessResult


class CorrectnessRunner(ABC):
    """Compare a compiled variant against a compiled baseline."""

    @abstractmethod
    async def compare(
        self,
        variant_id: str,
        baseline_variant_id: str,
        variant_result: CompilationResult,
        baseline_result: CompilationResult,
        tolerance_absolute: float = 1e-5,
        tolerance_relative: float = 1e-4,
    ) -> CorrectnessResult:
        """Return correctness comparison result for variant vs baseline."""
