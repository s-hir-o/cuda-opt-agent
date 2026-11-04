"""Tests for file-based correctness comparison."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from cuda_opt_agent.agents.benchmark_agent import BenchmarkAgent
from cuda_opt_agent.benchmark.mock_runner import MockBenchmarkRunner
from cuda_opt_agent.build.nvcc_runner import NVCCRunner
from cuda_opt_agent.core.config import BenchmarkConfig, CompilerConfig
from cuda_opt_agent.correctness.file_runner import FileOutputCorrectnessRunner
from cuda_opt_agent.schemas.compilation import CompilationResult


def _compilation(
    binary_path: str | None,
    success: bool = True,
) -> CompilationResult:
    return CompilationResult(
        success=success,
        source_code="__global__ void k() {}",
        arch="sm_80",
        binary_path=binary_path,
    )


class _MockProcess:
    def __init__(self, returncode: int = 0, stdout: bytes = b"", stderr: bytes = b""):
        self.returncode = returncode
        self._stdout = stdout
        self._stderr = stderr
        self.killed = False

    async def communicate(self):
        return self._stdout, self._stderr

    def kill(self):
        self.killed = True

    async def wait(self):
        return self.returncode


def _mock_exec_with_outputs(outputs: dict[str, np.ndarray], returncodes: dict[str, int] | None = None):
    async def _create_subprocess_exec(*args, **kwargs):
        binary_path = str(args[0])
        output_path = Path(args[-1])
        if binary_path in outputs:
            output_path.write_bytes(outputs[binary_path].tobytes())
        code = (returncodes or {}).get(binary_path, 0)
        return _MockProcess(returncode=code, stderr=b"execution failed" if code else b"")

    return _create_subprocess_exec


class TestFileOutputCorrectnessRunner:

    @pytest.mark.asyncio
    async def test_passes_for_matching_float_outputs(self):
        runner = FileOutputCorrectnessRunner(output_dtype="float32")
        outputs = {
            "baseline_exec": np.array([1.0, 2.0, 3.0], dtype=np.float32),
            "variant_exec": np.array([1.0, 2.000001, 3.0], dtype=np.float32),
        }

        with patch(
            "cuda_opt_agent.correctness.file_runner.asyncio.create_subprocess_exec",
            new=_mock_exec_with_outputs(outputs),
        ):
            result = await runner.compare(
                variant_id="v1",
                baseline_variant_id="baseline",
                variant_result=_compilation("variant_exec"),
                baseline_result=_compilation("baseline_exec"),
            )

        assert result.passed is True
        assert result.total_elements == 3
        assert result.metadata["runner"] == "file_output"
        assert result.metadata["contract"] == "--output <path>"

    @pytest.mark.asyncio
    async def test_fails_when_output_exceeds_tolerance(self):
        runner = FileOutputCorrectnessRunner(output_dtype="float32")
        outputs = {
            "baseline_exec": np.array([1.0, 2.0, 3.0], dtype=np.float32),
            "variant_exec": np.array([1.0, 2.5, 3.0], dtype=np.float32),
        }

        with patch(
            "cuda_opt_agent.correctness.file_runner.asyncio.create_subprocess_exec",
            new=_mock_exec_with_outputs(outputs),
        ):
            result = await runner.compare(
                variant_id="v1",
                baseline_variant_id="baseline",
                variant_result=_compilation("variant_exec"),
                baseline_result=_compilation("baseline_exec"),
                tolerance_absolute=1e-5,
                tolerance_relative=1e-4,
            )

        assert result.passed is False
        assert result.failed_elements == 1
        assert result.max_absolute_error == pytest.approx(0.5)
        assert "exceeded tolerance" in result.message

    @pytest.mark.asyncio
    async def test_passes_integer_outputs_only_when_exact(self):
        runner = FileOutputCorrectnessRunner(output_dtype="int32")
        outputs = {
            "baseline_exec": np.array([1, 2, 3], dtype=np.int32),
            "variant_exec": np.array([1, 2, 3], dtype=np.int32),
        }

        with patch(
            "cuda_opt_agent.correctness.file_runner.asyncio.create_subprocess_exec",
            new=_mock_exec_with_outputs(outputs),
        ):
            result = await runner.compare(
                variant_id="v1",
                baseline_variant_id="baseline",
                variant_result=_compilation("variant_exec"),
                baseline_result=_compilation("baseline_exec"),
            )

        assert result.passed is True

    @pytest.mark.asyncio
    async def test_fails_for_shape_mismatch(self):
        runner = FileOutputCorrectnessRunner(output_dtype="float32")
        outputs = {
            "baseline_exec": np.array([1.0, 2.0, 3.0], dtype=np.float32),
            "variant_exec": np.array([1.0, 2.0], dtype=np.float32),
        }

        with patch(
            "cuda_opt_agent.correctness.file_runner.asyncio.create_subprocess_exec",
            new=_mock_exec_with_outputs(outputs),
        ):
            result = await runner.compare(
                variant_id="v1",
                baseline_variant_id="baseline",
                variant_result=_compilation("variant_exec"),
                baseline_result=_compilation("baseline_exec"),
            )

        assert result.passed is False
        assert "Output shape mismatch" in result.message

    @pytest.mark.asyncio
    async def test_fails_when_variant_execution_fails(self):
        runner = FileOutputCorrectnessRunner(output_dtype="float32")
        outputs = {
            "baseline_exec": np.array([1.0], dtype=np.float32),
            "variant_exec": np.array([1.0], dtype=np.float32),
        }

        with patch(
            "cuda_opt_agent.correctness.file_runner.asyncio.create_subprocess_exec",
            new=_mock_exec_with_outputs(outputs, returncodes={"variant_exec": 2}),
        ):
            result = await runner.compare(
                variant_id="v1",
                baseline_variant_id="baseline",
                variant_result=_compilation("variant_exec"),
                baseline_result=_compilation("baseline_exec"),
            )

        assert result.passed is False
        assert "Variant correctness execution failed with code 2" in result.message
        assert result.metadata["variant_stderr"] == "execution failed"

    @pytest.mark.asyncio
    async def test_fails_without_runnable_binary_path(self):
        runner = FileOutputCorrectnessRunner(output_dtype="float32")

        result = await runner.compare(
            variant_id="v1",
            baseline_variant_id="baseline",
            variant_result=_compilation(None),
            baseline_result=_compilation("baseline_exec"),
        )

        assert result.passed is False
        assert "no runnable binary_path" in result.message

    def test_rejects_unsupported_dtype(self):
        with pytest.raises(ValueError, match="Unsupported correctness output_dtype"):
            FileOutputCorrectnessRunner(output_dtype="complex64")


class TestBenchmarkAgentFileCorrectness:

    @pytest.mark.asyncio
    async def test_forwards_configured_tolerances_to_file_runner(self):
        runner = FileOutputCorrectnessRunner(output_dtype="float32")
        outputs = {
            "baseline_exec": np.array([100.0], dtype=np.float32),
            "variant_exec": np.array([100.05], dtype=np.float32),
        }
        agent = BenchmarkAgent(
            nvcc_runner=NVCCRunner(CompilerConfig()),
            benchmark_runner=MockBenchmarkRunner(),
            benchmark_config=BenchmarkConfig(),
            correctness_runner=runner,
            correctness_tolerance_absolute=0.1,
            correctness_tolerance_relative=0.0,
        )

        with patch(
            "cuda_opt_agent.correctness.file_runner.asyncio.create_subprocess_exec",
            new=_mock_exec_with_outputs(outputs),
        ):
            result = await agent.check_correctness(
                variant_id="v1",
                baseline_id="baseline",
                variant_result=_compilation("variant_exec"),
                baseline_result=_compilation("baseline_exec"),
            )

        assert result.passed is True
        assert result.tolerance_absolute == 0.1
        assert result.tolerance_relative == 0.0
