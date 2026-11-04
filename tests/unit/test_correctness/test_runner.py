"""Tests for correctness runner interface and mock implementation."""

from __future__ import annotations

import pytest

from cuda_opt_agent.agents.benchmark_agent import BenchmarkAgent
from cuda_opt_agent.benchmark.mock_runner import MockBenchmarkRunner
from cuda_opt_agent.build.nvcc_runner import NVCCRunner
from cuda_opt_agent.core.config import BenchmarkConfig, CompilerConfig
from cuda_opt_agent.correctness.mock_runner import MockCorrectnessRunner
from cuda_opt_agent.schemas.compilation import CompilationResult


def _compilation(
    success: bool = True,
    binary_path: str | None = "/tmp/kernel_exec",
    cubin_path: str | None = None,
) -> CompilationResult:
    return CompilationResult(
        success=success,
        source_code="__global__ void k() {}",
        arch="sm_80",
        binary_path=binary_path,
        cubin_path=cubin_path,
    )


class TestMockCorrectnessRunner:

    @pytest.mark.asyncio
    async def test_passes_when_both_compilations_have_outputs(self):
        runner = MockCorrectnessRunner()

        result = await runner.compare(
            variant_id="v1",
            baseline_variant_id="baseline",
            variant_result=_compilation(binary_path="/tmp/v1"),
            baseline_result=_compilation(binary_path="/tmp/base"),
        )

        assert result.passed is True
        assert result.variant_id == "v1"
        assert result.baseline_variant_id == "baseline"
        assert result.message == "Mock correctness passed"

    @pytest.mark.asyncio
    async def test_accepts_cubin_paths(self):
        runner = MockCorrectnessRunner()

        result = await runner.compare(
            variant_id="v1",
            baseline_variant_id="baseline",
            variant_result=_compilation(binary_path=None, cubin_path="/tmp/v1.cubin"),
            baseline_result=_compilation(binary_path=None, cubin_path="/tmp/base.cubin"),
        )

        assert result.passed is True

    @pytest.mark.asyncio
    async def test_fails_when_variant_compilation_failed(self):
        runner = MockCorrectnessRunner()

        result = await runner.compare(
            variant_id="v1",
            baseline_variant_id="baseline",
            variant_result=_compilation(success=False),
            baseline_result=_compilation(),
        )

        assert result.passed is False
        assert "Variant compilation failed" in result.message

    @pytest.mark.asyncio
    async def test_fails_when_baseline_compilation_failed(self):
        runner = MockCorrectnessRunner()

        result = await runner.compare(
            variant_id="v1",
            baseline_variant_id="baseline",
            variant_result=_compilation(),
            baseline_result=_compilation(success=False),
        )

        assert result.passed is False
        assert "Baseline compilation failed" in result.message

    @pytest.mark.asyncio
    async def test_force_pass(self):
        runner = MockCorrectnessRunner(force_pass=True)

        result = await runner.compare(
            variant_id="v1",
            baseline_variant_id="baseline",
            variant_result=_compilation(success=False, binary_path=None),
            baseline_result=_compilation(success=False, binary_path=None),
        )

        assert result.passed is True
        assert result.metadata["forced"] is True

    @pytest.mark.asyncio
    async def test_force_fail(self):
        runner = MockCorrectnessRunner(force_pass=False)

        result = await runner.compare(
            variant_id="v1",
            baseline_variant_id="baseline",
            variant_result=_compilation(),
            baseline_result=_compilation(),
        )

        assert result.passed is False
        assert result.metadata["forced"] is True


class TestBenchmarkAgentCorrectness:

    @pytest.mark.asyncio
    async def test_delegates_to_runner_when_configured(self):
        agent = BenchmarkAgent(
            nvcc_runner=NVCCRunner(CompilerConfig()),
            benchmark_runner=MockBenchmarkRunner(),
            benchmark_config=BenchmarkConfig(),
            correctness_runner=MockCorrectnessRunner(force_pass=True),
        )

        result = await agent.check_correctness(
            variant_id="v1",
            baseline_id="baseline",
            variant_result=_compilation(),
            baseline_result=_compilation(),
        )

        assert result.passed is True
        assert result.metadata["runner"] == "mock"

    @pytest.mark.asyncio
    async def test_fails_closed_without_runner(self):
        agent = BenchmarkAgent(
            nvcc_runner=NVCCRunner(CompilerConfig()),
            benchmark_runner=MockBenchmarkRunner(),
            benchmark_config=BenchmarkConfig(),
        )

        result = await agent.check_correctness(
            variant_id="v1",
            baseline_id="baseline",
            variant_result=_compilation(),
            baseline_result=_compilation(),
        )

        assert result.passed is False
        assert "No correctness runner configured" in result.message
