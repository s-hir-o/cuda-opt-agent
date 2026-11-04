"""Deterministic mock correctness runner for tests and mock pipelines."""

from __future__ import annotations

from cuda_opt_agent.correctness.runner import CorrectnessRunner
from cuda_opt_agent.schemas.compilation import CompilationResult
from cuda_opt_agent.schemas.correctness import CorrectnessResult


class MockCorrectnessRunner(CorrectnessRunner):
    """Mock comparator with configurable pass/fail behavior.

    Default behavior passes only when both compilation results succeeded and
    both have either a runnable binary path or a cubin path.
    """

    def __init__(
        self,
        force_pass: bool | None = None,
        message: str | None = None,
    ) -> None:
        self._force_pass = force_pass
        self._message = message

    async def compare(
        self,
        variant_id: str,
        baseline_variant_id: str,
        variant_result: CompilationResult,
        baseline_result: CompilationResult,
        tolerance_absolute: float = 1e-5,
        tolerance_relative: float = 1e-4,
    ) -> CorrectnessResult:
        if self._force_pass is not None:
            return CorrectnessResult(
                passed=self._force_pass,
                variant_id=variant_id,
                baseline_variant_id=baseline_variant_id,
                tolerance_absolute=tolerance_absolute,
                tolerance_relative=tolerance_relative,
                message=self._message or _forced_message(self._force_pass),
                metadata={"runner": "mock", "forced": True},
            )

        if not variant_result.success:
            return _result(
                passed=False,
                variant_id=variant_id,
                baseline_variant_id=baseline_variant_id,
                tolerance_absolute=tolerance_absolute,
                tolerance_relative=tolerance_relative,
                message=self._message or "Variant compilation failed",
            )

        if not baseline_result.success:
            return _result(
                passed=False,
                variant_id=variant_id,
                baseline_variant_id=baseline_variant_id,
                tolerance_absolute=tolerance_absolute,
                tolerance_relative=tolerance_relative,
                message=self._message or "Baseline compilation failed",
            )

        if not _has_output_path(variant_result):
            return _result(
                passed=False,
                variant_id=variant_id,
                baseline_variant_id=baseline_variant_id,
                tolerance_absolute=tolerance_absolute,
                tolerance_relative=tolerance_relative,
                message=self._message or "Variant has no runnable binary or cubin path",
            )

        if not _has_output_path(baseline_result):
            return _result(
                passed=False,
                variant_id=variant_id,
                baseline_variant_id=baseline_variant_id,
                tolerance_absolute=tolerance_absolute,
                tolerance_relative=tolerance_relative,
                message=self._message or "Baseline has no runnable binary or cubin path",
            )

        return _result(
            passed=True,
            variant_id=variant_id,
            baseline_variant_id=baseline_variant_id,
            tolerance_absolute=tolerance_absolute,
            tolerance_relative=tolerance_relative,
            message=self._message or "Mock correctness passed",
        )


def _has_output_path(result: CompilationResult) -> bool:
    return bool(result.binary_path or result.cubin_path)


def _forced_message(force_pass: bool) -> str:
    return "Mock correctness forced pass" if force_pass else "Mock correctness forced fail"


def _result(
    passed: bool,
    variant_id: str,
    baseline_variant_id: str,
    tolerance_absolute: float,
    tolerance_relative: float,
    message: str,
) -> CorrectnessResult:
    return CorrectnessResult(
        passed=passed,
        variant_id=variant_id,
        baseline_variant_id=baseline_variant_id,
        tolerance_absolute=tolerance_absolute,
        tolerance_relative=tolerance_relative,
        message=message,
        metadata={"runner": "mock", "forced": False},
    )
