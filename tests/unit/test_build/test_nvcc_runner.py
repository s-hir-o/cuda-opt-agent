"""Tests for NVCCRunner compilation and diagnostic parsing."""

import os

import pytest

from cuda_opt_agent.build.nvcc_runner import NVCCRunner, parse_diagnostics
from cuda_opt_agent.core.config import CompilerConfig
from cuda_opt_agent.core.exceptions import PTXExtractionError
from cuda_opt_agent.schemas.compilation import DiagnosticSeverity
from cuda_opt_agent.schemas.kernel import KernelLanguage, KernelSource


SIMPLE_KERNEL = KernelSource(
    name="vecadd.cu",
    language=KernelLanguage.CUDA,
    source_code="__global__ void k() {}\n",
    entry_point="k",
)


# ── Diagnostic Parsing ─────────────────────────────────────────

class TestDiagnosticParsing:

    def test_standard_warning(self):
        stderr = 'vecadd.cu(3): warning: variable "x" was declared but never referenced'
        diags = parse_diagnostics(stderr)
        assert len(diags) == 1
        assert diags[0].severity == DiagnosticSeverity.WARNING
        assert diags[0].file == "vecadd.cu"
        assert diags[0].line == 3
        assert "never referenced" in diags[0].message

    def test_standard_error(self):
        stderr = 'vecadd.cu(5): error: identifier "foo" is undefined'
        diags = parse_diagnostics(stderr)
        assert len(diags) == 1
        assert diags[0].severity == DiagnosticSeverity.ERROR
        assert diags[0].line == 5

    def test_multiple_diagnostics(self):
        stderr = (
            'vecadd.cu(1): warning: unused var\n'
            'vecadd.cu(5): error: undefined\n'
            'vecadd.cu(10): remark: inlined\n'
        )
        diags = parse_diagnostics(stderr)
        assert len(diags) == 3
        severities = [d.severity for d in diags]
        assert severities == [
            DiagnosticSeverity.WARNING,
            DiagnosticSeverity.ERROR,
            DiagnosticSeverity.REMARK,
        ]

    def test_fatal_nvcc_error(self):
        stderr = "nvcc fatal   : Unsupported gpu architecture 'compute_99'"
        diags = parse_diagnostics(stderr)
        assert len(diags) == 1
        assert diags[0].severity == DiagnosticSeverity.ERROR
        assert "Unsupported" in diags[0].message

    def test_fatal_ptxas_error(self):
        stderr = "ptxas fatal   : Memory allocation failure"
        diags = parse_diagnostics(stderr)
        assert len(diags) == 1
        assert "Memory allocation" in diags[0].message

    def test_linker_error(self):
        stderr = "/usr/bin/ld: cannot find -lcudart"
        diags = parse_diagnostics(stderr)
        assert len(diags) == 1
        assert "cudart" in diags[0].message

    def test_no_false_positive_on_error_in_text(self):
        """Lines containing 'error' in file paths or comments must NOT match."""
        stderr = (
            "/home/user/error_handling/kernel.cu(10): warning: test\n"
            "// This is an error handling comment\n"
        )
        diags = parse_diagnostics(stderr)
        assert len(diags) == 1
        assert diags[0].severity == DiagnosticSeverity.WARNING

    def test_empty_stderr(self):
        assert parse_diagnostics("") == []

    def test_deduplicates(self):
        stderr = 'vecadd.cu(5): error: undefined\nvecadd.cu(5): error: undefined\n'
        diags = parse_diagnostics(stderr)
        assert len(diags) == 1

    def test_mixed_output(self):
        stderr = (
            'kernel.cu(12): warning: unused\n'
            'nvcc fatal   : Unsupported arch\n'
            "/usr/bin/ld: cannot find -lcudart\n"
            'kernel.cu(20): error: undefined\n'
        )
        diags = parse_diagnostics(stderr)
        assert len(diags) == 4
        severities = {d.severity for d in diags}
        assert DiagnosticSeverity.ERROR in severities
        assert DiagnosticSeverity.WARNING in severities

    def test_real_nvcc_warning_with_severity_code(self):
        """nvcc sometimes includes a warning code like '#177-D'."""
        stderr = '/tmp/kernel.cu(12): warning #177-D: variable "tmp" was declared but never referenced'
        diags = parse_diagnostics(stderr)
        assert len(diags) == 1
        assert diags[0].severity == DiagnosticSeverity.WARNING
        assert diags[0].line == 12
        assert "tmp" in diags[0].message

    def test_info_severity(self):
        stderr = "kernel.cu(7): info: compiler generated this code"
        diags = parse_diagnostics(stderr)
        assert len(diags) == 1
        assert diags[0].severity == DiagnosticSeverity.INFO


# ── Runner Initialization ──────────────────────────────────────

class TestNVCCRunnerInit:

    def test_default_config(self):
        config = CompilerConfig()
        runner = NVCCRunner(config)
        assert runner._nvcc_path == "nvcc"
        assert runner._default_arch == "sm_80"

    def test_custom_config(self):
        config = CompilerConfig(
            nvcc_path="/usr/local/cuda/bin/nvcc",
            default_arch="sm_89",
            extra_flags=["-DNDEBUG"],
        )
        runner = NVCCRunner(config)
        assert runner._nvcc_path == "/usr/local/cuda/bin/nvcc"
        assert runner._default_arch == "sm_89"

    def test_windows_uses_conforming_msvc_preprocessor(self):
        flags = NVCCRunner(CompilerConfig())._build_flags()

        assert ("-Xcompiler=/Zc:preprocessor" in flags) is (os.name == "nt")


# ── Availability ───────────────────────────────────────────────

class TestNVCCRunnerAvailability:

    @pytest.mark.asyncio
    async def test_availability_returns_bool(self):
        runner = NVCCRunner(CompilerConfig())
        assert isinstance(await runner.is_available(), bool)

    @pytest.mark.asyncio
    async def test_unavailable_with_bad_path(self):
        runner = NVCCRunner(CompilerConfig(nvcc_path="/nonexistent/nvcc"))
        assert await runner.is_available() is False


# ── extract_ptx interface ──────────────────────────────────────

class TestExtractPTX:

    @pytest.mark.asyncio
    async def test_extract_ptx_returns_string(self):
        """Interface contract: extract_ptx() -> str."""
        runner = NVCCRunner(CompilerConfig())
        if not await runner.is_available():
            pytest.skip("nvcc not available")
        ptx = await runner.extract_ptx(SIMPLE_KERNEL)
        assert isinstance(ptx, str)
        assert len(ptx) > 0

    @pytest.mark.asyncio
    async def test_extract_ptx_raises_on_bad_source(self):
        """Must raise PTXExtractionError on compilation failure."""
        runner = NVCCRunner(CompilerConfig())
        if not await runner.is_available():
            pytest.skip("nvcc not available")
        bad_kernel = KernelSource(
            name="bad.cu",
            language=KernelLanguage.CUDA,
            source_code="THIS IS NOT VALID CUDA",
        )
        with pytest.raises(PTXExtractionError):
            await runner.extract_ptx(bad_kernel)

    @pytest.mark.asyncio
    async def test_compile_ptx_returns_result(self):
        """compile_ptx() returns CompilationResult with diagnostics."""
        runner = NVCCRunner(CompilerConfig())
        if not await runner.is_available():
            pytest.skip("nvcc not available")
        result = await runner.compile_ptx(SIMPLE_KERNEL)
        assert result.arch == "sm_80"
        assert result.compile_time_s >= 0

    @pytest.mark.asyncio
    async def test_temp_dir_cleaned_up(self):
        """Temp dirs must not leak after extract_ptx."""
        import glob

        runner = NVCCRunner(CompilerConfig())
        if not await runner.is_available():
            pytest.skip("nvcc not available")

        before = set(glob.glob("/tmp/cuda_opt_*"))
        try:
            await runner.extract_ptx(SIMPLE_KERNEL)
        except PTXExtractionError:
            pass
        after = set(glob.glob("/tmp/cuda_opt_*"))
        leaked = after - before
        assert len(leaked) == 0, f"Leaked temp dirs: {leaked}"


# ── compile_executable binary_path ─────────────────────────────

class TestCompileExecutableBinaryPath:

    @pytest.mark.asyncio
    async def test_executable_sets_binary_path_not_cubin_path(self):
        """compile_executable() must set binary_path and NOT cubin_path."""
        runner = NVCCRunner(CompilerConfig())
        if not await runner.is_available():
            pytest.skip("nvcc not available")

        kernel_with_main = KernelSource(
            name="test_exec.cu",
            language=KernelLanguage.CUDA,
            source_code=(
                '#include <stdio.h>\n'
                '__global__ void k() {}\n'
                'int main() { k<<<1,1>>>(); cudaDeviceSynchronize(); return 0; }\n'
            ),
            entry_point="k",
        )
        result = await runner.compile_executable(kernel_with_main)
        if result.success:
            assert result.binary_path is not None
            assert result.cubin_path is None
