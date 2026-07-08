"""Mock-based integration tests for the full build pipeline.

These tests verify the compile -> parse -> store pipeline
without requiring a real nvcc installation.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cuda_opt_agent.build.arch_flags import common_flags
from cuda_opt_agent.build.artifact_manager import ArtifactManager
from cuda_opt_agent.build.nvcc_runner import NVCCRunner
from cuda_opt_agent.core.config import CompilerConfig
from cuda_opt_agent.core.exceptions import ConfigurationError
from cuda_opt_agent.memory.artifact_store import ArtifactStore
from cuda_opt_agent.schemas.compilation import CompilationResult, DiagnosticSeverity
from cuda_opt_agent.schemas.kernel import KernelLanguage, KernelSource


MOCK_KERNEL = KernelSource(
    name="test.cu",
    language=KernelLanguage.CUDA,
    source_code="__global__ void vecadd() { /* optimized */ }\n",
    entry_point="vecadd",
)


def _mock_nvcc_process(
    returncode: int = 0,
    stdout: bytes = b"",
    stderr: bytes = b"",
    write_output: bool = True,
):
    """Create a mock async subprocess that simulates nvcc behavior."""

    async def _communicate():
        return stdout, stderr

    proc = AsyncMock()
    proc.communicate = _communicate
    proc.returncode = returncode

    async def _create_subprocess(*args, **kwargs):
        if write_output:
            # Extract output_path from command args
            cmd = list(args)
            if "-o" in cmd:
                idx = cmd.index("-o")
                if idx + 1 < len(cmd):
                    out_path = Path(cmd[idx + 1])
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    # Write a dummy output file
                    if str(out_path).endswith(".ptx"):
                        out_path.write_text("// PTX generated\n")
                    else:
                        out_path.write_bytes(b"\x7fELF\x00")
        return proc

    return _create_subprocess


class TestMockCompileToCubin:

    @pytest.mark.asyncio
    async def test_compile_success(self, tmp_path: Path):
        """Full pipeline: compile kernel -> get CompilationResult with cubin."""
        runner = NVCCRunner(CompilerConfig())

        with patch(
            "cuda_opt_agent.build.nvcc_runner.asyncio.create_subprocess_exec",
            new=_mock_nvcc_process(returncode=0),
        ):
            result = await runner.compile(
                MOCK_KERNEL,
                arch="sm_80",
                output_dir=str(tmp_path / "out"),
            )

        assert result.success is True
        assert result.cubin_path is not None
        assert result.binary_path is None  # cubin compile does NOT set binary_path
        assert result.arch == "sm_80"
        assert result.compile_time_s >= 0

    @pytest.mark.asyncio
    async def test_compile_failure(self, tmp_path: Path):
        runner = NVCCRunner(CompilerConfig())

        with patch(
            "cuda_opt_agent.build.nvcc_runner.asyncio.create_subprocess_exec",
            new=_mock_nvcc_process(
                returncode=1,
                stderr=b"kernel.cu(3): error: undefined identifier 'foo'\n",
            ),
        ):
            result = await runner.compile(
                MOCK_KERNEL,
                output_dir=str(tmp_path / "out"),
            )

        assert result.success is False
        assert result.cubin_path is None
        assert len(result.diagnostics) >= 1
        assert any(d.severity == DiagnosticSeverity.ERROR for d in result.diagnostics)


class TestMockExtractPTX:

    @pytest.mark.asyncio
    async def test_extract_ptx_success(self, tmp_path: Path):
        """Full pipeline: extract PTX -> string returned."""
        runner = NVCCRunner(CompilerConfig())

        with patch(
            "cuda_opt_agent.build.nvcc_runner.asyncio.create_subprocess_exec",
            new=_mock_nvcc_process(returncode=0),
        ):
            ptx = await runner.extract_ptx(MOCK_KERNEL)

        assert isinstance(ptx, str)
        assert "PTX" in ptx

    @pytest.mark.asyncio
    async def test_extract_ptx_failure_raises(self, tmp_path: Path):
        runner = NVCCRunner(CompilerConfig())

        with patch(
            "cuda_opt_agent.build.nvcc_runner.asyncio.create_subprocess_exec",
            new=_mock_nvcc_process(
                returncode=1,
                stderr=b"kernel.cu(1): error: syntax error\n",
            ),
        ):
            from cuda_opt_agent.core.exceptions import PTXExtractionError
            with pytest.raises(PTXExtractionError):
                await runner.extract_ptx(MOCK_KERNEL)


class TestMockCompileExecutable:

    @pytest.mark.asyncio
    async def test_compile_executable_sets_binary_path(self, tmp_path: Path):
        """compile_executable() must set binary_path, NOT cubin_path."""
        runner = NVCCRunner(CompilerConfig())

        with patch(
            "cuda_opt_agent.build.nvcc_runner.asyncio.create_subprocess_exec",
            new=_mock_nvcc_process(returncode=0),
        ):
            result = await runner.compile_executable(
                MOCK_KERNEL,
                output_dir=str(tmp_path / "out"),
            )

        assert result.success is True
        assert result.binary_path is not None
        assert result.cubin_path is None


class TestMockFullPipeline:
    """End-to-end: compile -> parse -> store -> retrieve."""

    @pytest.mark.asyncio
    async def test_compile_and_store_cubin(self, tmp_path: Path):
        runner = NVCCRunner(CompilerConfig())
        store = ArtifactStore(tmp_path / "artifacts")
        mgr = ArtifactManager(store, run_id="run_001")

        with patch(
            "cuda_opt_agent.build.nvcc_runner.asyncio.create_subprocess_exec",
            new=_mock_nvcc_process(
                returncode=0,
                stderr=b"kernel.cu(2): warning: unused variable\n",
            ),
        ):
            result = await runner.compile(
                MOCK_KERNEL,
                output_dir=str(tmp_path / "compile_out"),
            )

        assert result.success is True

        # Store all artifacts
        paths = mgr.store_compilation_result("v1", result)
        assert len(paths) >= 1

        # Verify retrieval
        source = mgr.get_source("v1")
        assert source is not None
        assert "vecadd" in source

        # Verify records
        records = mgr.get_records("v1")
        kinds = {r.kind for r in records}
        assert "source" in kinds
        assert "cubin" in kinds

        # Verify diagnostics were parsed
        assert len(result.diagnostics) >= 1
        assert result.diagnostics[0].severity == DiagnosticSeverity.WARNING

    @pytest.mark.asyncio
    async def test_compile_ptx_and_store(self, tmp_path: Path):
        runner = NVCCRunner(CompilerConfig())
        store = ArtifactStore(tmp_path / "artifacts")
        mgr = ArtifactManager(store, run_id="run_002")

        with patch(
            "cuda_opt_agent.build.nvcc_runner.asyncio.create_subprocess_exec",
            new=_mock_nvcc_process(returncode=0),
        ):
            result = await runner.compile_ptx(
                MOCK_KERNEL,
                output_dir=str(tmp_path / "compile_out"),
            )

        assert result.success is True
        assert result.ptx_source is not None

        paths = mgr.store_compilation_result("v1", result)
        assert mgr.get_ptx("v1") is not None
        kinds = {r.kind for r in mgr.get_records("v1")}
        assert "ptx" in kinds

    @pytest.mark.asyncio
    async def test_compile_executable_and_store(self, tmp_path: Path):
        runner = NVCCRunner(CompilerConfig())
        store = ArtifactStore(tmp_path / "artifacts")
        mgr = ArtifactManager(store, run_id="run_003")

        with patch(
            "cuda_opt_agent.build.nvcc_runner.asyncio.create_subprocess_exec",
            new=_mock_nvcc_process(returncode=0),
        ):
            result = await runner.compile_executable(
                MOCK_KERNEL,
                output_dir=str(tmp_path / "compile_out"),
            )

        assert result.success is True
        assert result.binary_path is not None
        assert result.cubin_path is None

        # store_compilation_result should detect binary_path and store as executable
        paths = mgr.store_compilation_result("v1", result)
        kinds = {r.kind for r in mgr.get_records("v1")}
        assert "executable" in kinds
        assert "source" in kinds
        # cubin should NOT be present
        assert "cubin" not in kinds

        exe_path = mgr.get_executable_path("v1")
        assert exe_path is not None

    @pytest.mark.asyncio
    async def test_full_pipeline_with_cleanup(self, tmp_path: Path):
        runner = NVCCRunner(CompilerConfig())
        store = ArtifactStore(tmp_path / "artifacts")
        mgr = ArtifactManager(store, run_id="run_cleanup")

        with patch(
            "cuda_opt_agent.build.nvcc_runner.asyncio.create_subprocess_exec",
            new=_mock_nvcc_process(returncode=0),
        ):
            # Compile two variants
            r1 = await runner.compile(MOCK_KERNEL, output_dir=str(tmp_path / "v1"))
            r2 = await runner.compile(MOCK_KERNEL, output_dir=str(tmp_path / "v2"))

        mgr.store_compilation_result("v1", r1)
        mgr.store_compilation_result("v2", r2)
        assert mgr.variant_count() == 2

        # Cleanup variant 1
        mgr.cleanup_variant("v1")
        assert mgr.variant_count() == 1
        assert mgr.get_source("v1") is None
        assert mgr.get_source("v2") is not None

        # Cleanup entire run
        mgr.cleanup_run()
        assert mgr.variant_count() == 0
        assert mgr.total_size_bytes() == 0


class TestFlagsIntegration:

    def test_compile_flag_combinations(self):
        """Verify common_flags integrates correctly for all optimization levels."""
        for level in range(4):
            flags = common_flags(
                optimization_level=level,
                fast_math=True,
                lineinfo=True,
                extra=["-DNDEBUG"],
            )
            assert f"-O{level}" in flags
            assert "--use_fast_math" in flags
            assert "-lineinfo" in flags
            assert "-DNDEBUG" in flags

    def test_invalid_level_in_pipeline_raises(self):
        """Bad optimization level should fail before reaching nvcc."""
        with pytest.raises(ConfigurationError):
            common_flags(optimization_level=99)
