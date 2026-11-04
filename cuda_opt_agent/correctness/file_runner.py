"""File-based correctness runner for host-runnable CUDA executables.

Contract:
    executable --output <path>

Both baseline and variant executables must write the same raw numeric output
buffer to the supplied path. This runner compares those buffers element-wise.
"""

from __future__ import annotations

import asyncio
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from cuda_opt_agent.correctness.runner import CorrectnessRunner
from cuda_opt_agent.schemas.compilation import CompilationResult
from cuda_opt_agent.schemas.correctness import CorrectnessResult


_SUPPORTED_DTYPES = {
    "float32",
    "float64",
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
}


@dataclass(frozen=True)
class _ExecutionResult:
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool = False


class FileOutputCorrectnessRunner(CorrectnessRunner):
    """Compare executable outputs written through a file contract.

    The runner intentionally requires ``CompilationResult.binary_path``. A
    cubin is a device binary and cannot be executed as a host process.
    """

    def __init__(
        self,
        output_dtype: str = "float32",
        output_count: int | None = None,
        timeout_seconds: float = 120.0,
        output_flag: str = "--output",
        extra_args: list[str] | None = None,
    ) -> None:
        if output_dtype not in _SUPPORTED_DTYPES:
            supported = ", ".join(sorted(_SUPPORTED_DTYPES))
            raise ValueError(
                f"Unsupported correctness output_dtype '{output_dtype}'. "
                f"Supported dtypes: {supported}"
            )
        if output_count is not None and output_count <= 0:
            raise ValueError("output_count must be positive when provided")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")

        self._dtype = np.dtype(output_dtype)
        self._output_count = output_count
        self._timeout_seconds = timeout_seconds
        self._output_flag = output_flag
        self._extra_args = list(extra_args or [])

    async def compare(
        self,
        variant_id: str,
        baseline_variant_id: str,
        variant_result: CompilationResult,
        baseline_result: CompilationResult,
        tolerance_absolute: float = 1e-5,
        tolerance_relative: float = 1e-4,
    ) -> CorrectnessResult:
        if not variant_result.success:
            return self._failure(
                variant_id,
                baseline_variant_id,
                tolerance_absolute,
                tolerance_relative,
                "Variant compilation failed",
            )

        if not baseline_result.success:
            return self._failure(
                variant_id,
                baseline_variant_id,
                tolerance_absolute,
                tolerance_relative,
                "Baseline compilation failed",
            )

        if not variant_result.binary_path:
            return self._failure(
                variant_id,
                baseline_variant_id,
                tolerance_absolute,
                tolerance_relative,
                "Variant has no runnable binary_path for file correctness",
            )

        if not baseline_result.binary_path:
            return self._failure(
                variant_id,
                baseline_variant_id,
                tolerance_absolute,
                tolerance_relative,
                "Baseline has no runnable binary_path for file correctness",
            )

        with tempfile.TemporaryDirectory(prefix="cuda_correctness_") as tmp:
            tmp_dir = Path(tmp)
            baseline_output = tmp_dir / "baseline.bin"
            variant_output = tmp_dir / "variant.bin"

            baseline_exec = await self._run_binary(
                baseline_result.binary_path,
                baseline_output,
            )
            if baseline_exec.returncode != 0:
                return self._execution_failure(
                    variant_id,
                    baseline_variant_id,
                    tolerance_absolute,
                    tolerance_relative,
                    "baseline",
                    baseline_exec,
                )

            variant_exec = await self._run_binary(
                variant_result.binary_path,
                variant_output,
            )
            if variant_exec.returncode != 0:
                return self._execution_failure(
                    variant_id,
                    baseline_variant_id,
                    tolerance_absolute,
                    tolerance_relative,
                    "variant",
                    variant_exec,
                )

            try:
                baseline_values = self._read_output(baseline_output)
                variant_values = self._read_output(variant_output)
            except ValueError as exc:
                return self._failure(
                    variant_id,
                    baseline_variant_id,
                    tolerance_absolute,
                    tolerance_relative,
                    str(exc),
                )

            return self._compare_arrays(
                variant_id=variant_id,
                baseline_variant_id=baseline_variant_id,
                variant_values=variant_values,
                baseline_values=baseline_values,
                tolerance_absolute=tolerance_absolute,
                tolerance_relative=tolerance_relative,
            )

    async def _run_binary(self, binary_path: str, output_path: Path) -> _ExecutionResult:
        cmd = [
            binary_path,
            *self._extra_args,
            self._output_flag,
            str(output_path),
        ]

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=self._timeout_seconds,
            )
        except FileNotFoundError:
            return _ExecutionResult(
                returncode=-1,
                stdout="",
                stderr=f"Executable not found: {binary_path}",
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return _ExecutionResult(
                returncode=-1,
                stdout="",
                stderr=f"Execution timed out after {self._timeout_seconds:g} seconds",
                timed_out=True,
            )

        return _ExecutionResult(
            returncode=proc.returncode or 0,
            stdout=stdout.decode(errors="replace"),
            stderr=stderr.decode(errors="replace"),
        )

    def _read_output(self, output_path: Path) -> np.ndarray:
        if not output_path.is_file():
            raise ValueError(
                f"Executable did not create correctness output file: {output_path.name}"
            )

        raw = output_path.read_bytes()
        if not raw:
            raise ValueError(
                f"Executable created empty correctness output file: {output_path.name}"
            )

        if len(raw) % self._dtype.itemsize != 0:
            raise ValueError(
                f"Output byte count {len(raw)} is not divisible by dtype "
                f"size {self._dtype.itemsize} for {self._dtype.name}"
            )

        values = np.frombuffer(raw, dtype=self._dtype)
        if self._output_count is not None and values.size != self._output_count:
            raise ValueError(
                f"Expected {self._output_count} output elements, got {values.size}"
            )
        return values

    def _compare_arrays(
        self,
        variant_id: str,
        baseline_variant_id: str,
        variant_values: np.ndarray,
        baseline_values: np.ndarray,
        tolerance_absolute: float,
        tolerance_relative: float,
    ) -> CorrectnessResult:
        if variant_values.shape != baseline_values.shape:
            return self._failure(
                variant_id,
                baseline_variant_id,
                tolerance_absolute,
                tolerance_relative,
                (
                    "Output shape mismatch: "
                    f"variant={variant_values.shape}, baseline={baseline_values.shape}"
                ),
                total_elements=max(variant_values.size, baseline_values.size),
            )

        if np.issubdtype(self._dtype, np.floating):
            return self._compare_floating(
                variant_id,
                baseline_variant_id,
                variant_values,
                baseline_values,
                tolerance_absolute,
                tolerance_relative,
            )

        mismatches = variant_values != baseline_values
        failed_elements = int(np.count_nonzero(mismatches))
        passed = failed_elements == 0
        max_abs = float(
            np.max(
                np.abs(
                    variant_values.astype(np.int64) - baseline_values.astype(np.int64)
                )
            )
        ) if variant_values.size else 0.0

        return CorrectnessResult(
            passed=passed,
            variant_id=variant_id,
            baseline_variant_id=baseline_variant_id,
            max_absolute_error=max_abs,
            max_relative_error=0.0,
            tolerance_absolute=tolerance_absolute,
            tolerance_relative=tolerance_relative,
            failed_elements=failed_elements,
            total_elements=int(variant_values.size),
            message="File correctness passed" if passed else "Integer outputs differ",
            metadata=self._metadata(),
        )

    def _compare_floating(
        self,
        variant_id: str,
        baseline_variant_id: str,
        variant_values: np.ndarray,
        baseline_values: np.ndarray,
        tolerance_absolute: float,
        tolerance_relative: float,
    ) -> CorrectnessResult:
        abs_error = np.abs(variant_values - baseline_values)
        denominator = np.maximum(np.abs(baseline_values), tolerance_absolute)
        rel_error = abs_error / denominator
        allowed_error = tolerance_absolute + tolerance_relative * np.abs(baseline_values)
        failures = abs_error > allowed_error
        failed_elements = int(np.count_nonzero(failures))
        passed = failed_elements == 0

        return CorrectnessResult(
            passed=passed,
            variant_id=variant_id,
            baseline_variant_id=baseline_variant_id,
            max_absolute_error=float(np.max(abs_error)) if abs_error.size else 0.0,
            max_relative_error=float(np.max(rel_error)) if rel_error.size else 0.0,
            tolerance_absolute=tolerance_absolute,
            tolerance_relative=tolerance_relative,
            failed_elements=failed_elements,
            total_elements=int(variant_values.size),
            message=(
                "File correctness passed"
                if passed
                else f"{failed_elements} output elements exceeded tolerance"
            ),
            metadata=self._metadata(),
        )

    def _execution_failure(
        self,
        variant_id: str,
        baseline_variant_id: str,
        tolerance_absolute: float,
        tolerance_relative: float,
        role: str,
        result: _ExecutionResult,
    ) -> CorrectnessResult:
        if result.timed_out:
            message = f"{role.capitalize()} correctness execution timed out"
        else:
            stderr = result.stderr.strip()[:500]
            message = (
                f"{role.capitalize()} correctness execution failed with "
                f"code {result.returncode}: {stderr}"
            )

        metadata = self._metadata()
        metadata.update(
            {
                f"{role}_returncode": result.returncode,
                f"{role}_stdout": result.stdout.strip()[:500],
                f"{role}_stderr": result.stderr.strip()[:500],
            }
        )
        return CorrectnessResult(
            passed=False,
            variant_id=variant_id,
            baseline_variant_id=baseline_variant_id,
            tolerance_absolute=tolerance_absolute,
            tolerance_relative=tolerance_relative,
            message=message,
            metadata=metadata,
        )

    def _failure(
        self,
        variant_id: str,
        baseline_variant_id: str,
        tolerance_absolute: float,
        tolerance_relative: float,
        message: str,
        total_elements: int = 0,
    ) -> CorrectnessResult:
        return CorrectnessResult(
            passed=False,
            variant_id=variant_id,
            baseline_variant_id=baseline_variant_id,
            tolerance_absolute=tolerance_absolute,
            tolerance_relative=tolerance_relative,
            total_elements=total_elements,
            message=message,
            metadata=self._metadata(),
        )

    def _metadata(self) -> dict[str, object]:
        return {
            "runner": "file_output",
            "contract": f"{self._output_flag} <path>",
            "dtype": self._dtype.name,
            "output_count": self._output_count,
            "timeout_seconds": self._timeout_seconds,
        }
