"""Tests for ArtifactManager."""

import pytest
from pathlib import Path

from cuda_opt_agent.build.artifact_manager import ArtifactManager
from cuda_opt_agent.memory.artifact_store import ArtifactStore
from cuda_opt_agent.schemas.compilation import CompilationResult


class TestArtifactManager:

    @pytest.fixture
    def manager(self, tmp_path: Path) -> ArtifactManager:
        store = ArtifactStore(tmp_path / "artifacts")
        return ArtifactManager(store, run_id="run_test")

    @pytest.fixture
    def sample_result(self) -> CompilationResult:
        return CompilationResult(
            success=True,
            source_code="__global__ void k() {}",
            arch="sm_80",
            ptx_source="// PTX output",
            compile_time_s=1.5,
        )

    @pytest.fixture
    def executable_result(self, tmp_path: Path) -> CompilationResult:
        exe = tmp_path / "kernel_exec"
        exe.write_bytes(b"\x7fELF\x00\x00")
        return CompilationResult(
            success=True,
            source_code="int main() {}",
            arch="sm_80",
            binary_path=str(exe),
            compile_time_s=2.0,
        )

    # ── Source ─────────────────────────────────────────────────

    def test_store_source(self, manager: ArtifactManager):
        rel = manager.store_source("v1", "// kernel source")
        assert "run_test" in rel
        assert manager.get_source("v1") == "// kernel source"

    def test_get_source_not_found(self, manager: ArtifactManager):
        assert manager.get_source("nonexistent") is None

    # ── PTX ────────────────────────────────────────────────────

    def test_store_ptx_standalone(self, manager: ArtifactManager):
        rel = manager.store_ptx("v1", "// ptx standalone")
        assert "ptx" in rel
        assert manager.get_ptx("v1") == "// ptx standalone"

    def test_get_ptx_not_found(self, manager: ArtifactManager):
        assert manager.get_ptx("nonexistent") is None

    # ── Cubin ──────────────────────────────────────────────────

    def test_store_cubin_missing_file(self, manager: ArtifactManager):
        result = manager.store_cubin("v1", "/nonexistent/file.cubin")
        assert result is None

    # ── Executable ─────────────────────────────────────────────

    def test_store_executable(self, manager: ArtifactManager, tmp_path: Path):
        exe = tmp_path / "kernel_exec"
        exe.write_bytes(b"\x7fELF")
        rel = manager.store_executable("v1", str(exe))
        assert rel is not None
        assert manager.get_executable_path("v1") is not None

    def test_store_executable_missing_file(self, manager: ArtifactManager):
        assert manager.store_executable("v1", "/nonexistent/exec") is None

    def test_get_executable_path_none(self, manager: ArtifactManager):
        assert manager.get_executable_path("v1") is None

    # ── CompilationResult integration ──────────────────────────

    def test_store_compilation_result(self, manager: ArtifactManager, sample_result: CompilationResult):
        paths = manager.store_compilation_result("v1", sample_result)
        assert len(paths) >= 1
        assert manager.get_ptx("v1") == "// PTX output"

    def test_store_compilation_result_with_binary_path(
        self, manager: ArtifactManager, executable_result: CompilationResult
    ):
        """binary_path result must be stored as kind='executable', not 'cubin'."""
        paths = manager.store_compilation_result("v1", executable_result)
        kinds = {r.kind for r in manager.get_records("v1")}
        assert "executable" in kinds
        assert "cubin" not in kinds
        assert manager.get_executable_path("v1") is not None

    def test_store_compilation_result_no_cubin_no_binary(
        self, manager: ArtifactManager, sample_result: CompilationResult
    ):
        """Result with only source + PTX — no cubin or executable records."""
        manager.store_compilation_result("v1", sample_result)
        kinds = {r.kind for r in manager.get_records("v1")}
        assert "source" in kinds
        assert "ptx" in kinds
        assert "cubin" not in kinds
        assert "executable" not in kinds

    # ── Records ────────────────────────────────────────────────

    def test_get_records(self, manager: ArtifactManager, sample_result: CompilationResult):
        manager.store_compilation_result("v1", sample_result)
        records = manager.get_records("v1")
        kinds = {r.kind for r in records}
        assert "source" in kinds
        assert "ptx" in kinds

    # ── Counts ─────────────────────────────────────────────────

    def test_variant_count(self, manager: ArtifactManager):
        manager.store_source("v1", "a")
        manager.store_source("v2", "b")
        assert manager.variant_count() == 2

    def test_artifact_count(self, manager: ArtifactManager, sample_result: CompilationResult):
        manager.store_compilation_result("v1", sample_result)
        assert manager.artifact_count() >= 2

    # ── Cleanup ────────────────────────────────────────────────

    def test_cleanup_variant(self, manager: ArtifactManager):
        manager.store_source("v1", "a")
        manager.store_source("v2", "b")
        manager.cleanup_variant("v1")
        assert manager.get_source("v1") is None
        assert manager.get_source("v2") is not None
        assert manager.variant_count() == 1

    def test_cleanup_run(self, manager: ArtifactManager):
        manager.store_source("v1", "a")
        manager.store_source("v2", "b")
        manager.cleanup_run()
        assert manager.variant_count() == 0
        assert manager.artifact_count() == 0

    def test_cleanup_variant_nonexistent_is_noop(self, manager: ArtifactManager):
        manager.cleanup_variant("nonexistent")  # Should not raise
        assert manager.variant_count() == 0

    # ── Reporting ──────────────────────────────────────────────

    def test_summary(self, manager: ArtifactManager, sample_result: CompilationResult):
        manager.store_compilation_result("v1", sample_result)
        s = manager.summary()
        assert s["run_id"] == "run_test"
        assert s["variant_count"] == 1
        assert "v1" in s["variants"]

    def test_total_size_bytes(self, manager: ArtifactManager):
        manager.store_source("v1", "// hello world")
        assert manager.total_size_bytes() > 0

    # ── Isolation ──────────────────────────────────────────────

    def test_multiple_variants_isolation(self, manager: ArtifactManager):
        manager.store_source("v1", "source v1")
        manager.store_source("v2", "source v2")
        assert manager.get_source("v1") == "source v1"
        assert manager.get_source("v2") == "source v2"
        assert manager.get_records("v1") != manager.get_records("v2")
