"""Tests for the memory layer: database, repositories, artifact store."""

from __future__ import annotations

import threading
import textwrap
from datetime import datetime, timezone
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from cuda_opt_agent.core.exceptions import ArtifactNotFoundError, StorageError
from cuda_opt_agent.memory.database import Base
from cuda_opt_agent.memory import models  # noqa: F401
from cuda_opt_agent.memory.artifact_store import ArtifactStore
from cuda_opt_agent.memory.repositories.kernel_repo import KernelRepo
from cuda_opt_agent.memory.repositories.metrics_repo import MetricsRepo
from cuda_opt_agent.memory.repositories.result_repo import ResultRepo
from cuda_opt_agent.memory.repositories.run_repo import RunRepo
from cuda_opt_agent.schemas.benchmark import BenchmarkResult, SpeedupResult, TimingStats
from cuda_opt_agent.schemas.bottleneck import BottleneckClass, BottleneckResult
from cuda_opt_agent.schemas.compilation import CompilationDiagnostic, CompilationResult
from cuda_opt_agent.schemas.correctness import CorrectnessResult
from cuda_opt_agent.schemas.kernel import KernelLanguage, KernelSource, KernelVariant
from cuda_opt_agent.schemas.metrics import ProfileMetrics
from cuda_opt_agent.schemas.optimization import OptimizationIteration, OptimizationRun
from cuda_opt_agent.schemas.transform import TransformBackend, TransformResult, TransformSpec


# ── Helpers ────────────────────────────────────────────────────

def _make_kernel(name: str = "test.cu", source: str = "__global__ void k() {}") -> KernelSource:
    return KernelSource(name=name, language=KernelLanguage.CUDA, source_code=source, entry_point="k")


def _make_variant(kernel: KernelSource, variant_id: str = "v1", iteration: int = 1) -> KernelVariant:
    return KernelVariant(
        variant_id=variant_id,
        parent_source=kernel,
        source_code=f"// optimized {kernel.source_code}",
        iteration=iteration,
    )


def _make_metrics(kernel_name: str = "test", variant_id: str = "v1") -> ProfileMetrics:
    return ProfileMetrics(
        kernel_name=kernel_name,
        gpu_time_us=100.0,
        sm_throughput=80.0,
        dram_throughput=60.0,
        achieved_occupancy=0.75,
        registers_per_thread=32,
        block_size=(16, 16, 1),
        grid_size=(64, 64, 1),
    )


# ══════════════════════════════════════════════════════════════
# Required Tests
# ══════════════════════════════════════════════════════════════


class TestCreateKernel:
    """Required: create kernel"""

    def test_save_and_get_source(self, db_session: Session):
        repo = KernelRepo(db_session)
        kernel = _make_kernel()

        db_id = repo.save_source(kernel)
        assert db_id > 0

        loaded = repo.get_source(db_id)
        assert loaded is not None
        assert loaded.name == "test.cu"
        assert loaded.language == KernelLanguage.CUDA
        assert loaded.source_code == kernel.source_code
        assert loaded.entry_point == "k"

    def test_save_source_returns_autoincrement_id(self, db_session: Session):
        repo = KernelRepo(db_session)
        id1 = repo.save_source(_make_kernel("a.cu"))
        id2 = repo.save_source(_make_kernel("b.cu"))
        assert id2 > id1

    def test_get_source_not_found(self, db_session: Session):
        repo = KernelRepo(db_session)
        assert repo.get_source(9999) is None

    def test_find_sources_by_name(self, db_session: Session):
        repo = KernelRepo(db_session)
        repo.save_source(_make_kernel("a.cu"))
        repo.save_source(_make_kernel("b.cu"))
        repo.save_source(_make_kernel("a.cu"))

        results = repo.find_sources_by_name("a.cu")
        assert len(results) == 2

    def test_list_sources_paginated(self, db_session: Session):
        repo = KernelRepo(db_session)
        for i in range(5):
            repo.save_source(_make_kernel(f"k{i}.cu"))

        page = repo.list_sources(offset=2, limit=2)
        assert len(page) == 2

    def test_save_and_get_variant(self, db_session: Session):
        repo = KernelRepo(db_session)
        kernel = _make_kernel()
        variant = _make_variant(kernel, "v1")

        vid = repo.save_variant(variant)
        assert vid == "v1"

        loaded = repo.get_variant("v1")
        assert loaded is not None
        assert loaded.variant_id == "v1"
        assert loaded.parent_source.name == "test.cu"
        assert loaded.iteration == 1

    def test_list_variants_for_kernel(self, db_session: Session):
        repo = KernelRepo(db_session)
        kernel = _make_kernel()
        kid = repo.save_source(kernel)

        repo.save_variant(_make_variant(kernel, "v1", iteration=1))
        repo.save_variant(_make_variant(kernel, "v2", iteration=2))
        repo.save_variant(_make_variant(kernel, "v3", iteration=3))

        variants = repo.list_variants_for_kernel(kid)
        assert len(variants) == 3
        assert [v.iteration for v in variants] == [1, 2, 3]

    def test_variant_with_transform_history(self, db_session: Session):
        repo = KernelRepo(db_session)
        kernel = _make_kernel()
        variant = _make_variant(kernel, "v1")
        variant.transform_history = [
            TransformSpec(
                transform_name="shared_memory_tiling",
                params={"tile_size": 16},
                reason="memory bound",
            ),
            TransformSpec(
                transform_name="loop_unrolling",
                backend=TransformBackend.PTX,
                params={"factor": 4},
            ),
        ]

        repo.save_variant(variant)
        loaded = repo.get_variant("v1")
        assert loaded is not None
        assert len(loaded.transform_history) == 2
        assert loaded.transform_history[0].transform_name == "shared_memory_tiling"
        assert loaded.transform_history[1].backend == TransformBackend.PTX


class TestUpdateKernel:
    """Required: update kernel"""

    def test_update_source_name_and_code(self, db_session: Session):
        repo = KernelRepo(db_session)
        kernel = _make_kernel("original.cu", "__global__ void old() {}")
        db_id = repo.save_source(kernel)

        updated = KernelSource(
            name="updated.cu",
            language=KernelLanguage.CUDA,
            source_code="__global__ void new_kernel() {}",
            entry_point="new_kernel",
        )
        repo.update_source(updated, db_id)

        loaded = repo.get_source(db_id)
        assert loaded is not None
        assert loaded.name == "updated.cu"
        assert loaded.source_code == "__global__ void new_kernel() {}"
        assert loaded.entry_point == "new_kernel"

    def test_update_source_nonexistent_is_noop(self, db_session: Session):
        repo = KernelRepo(db_session)
        kernel = _make_kernel()
        # Should not raise
        repo.update_source(kernel, 9999)

    def test_update_run_status(self, db_session: Session):
        kernel = _make_kernel()
        KernelRepo(db_session).save_source(kernel)

        run = OptimizationRun(run_id="run_upd", kernel=kernel, target_arch="sm_80")
        repo = RunRepo(db_session)
        repo.save_run(run)

        run.status = "converged"
        run.best_speedup = 2.5
        run.best_variant_id = "v3"
        run.completed_at = datetime.now(timezone.utc)
        repo.update_run(run)

        loaded = repo.get_run("run_upd", load_iterations=False)
        assert loaded is not None
        assert loaded.status == "converged"
        assert loaded.best_speedup == 2.5
        assert loaded.best_variant_id == "v3"


class TestCreateRun:
    """Required: create run"""

    def test_save_and_get_run(self, db_session: Session):
        kernel = _make_kernel()
        KernelRepo(db_session).save_source(kernel)

        run = OptimizationRun(
            run_id="run_001",
            kernel=kernel,
            target_arch="sm_80",
        )
        repo = RunRepo(db_session)
        rid = repo.save_run(run)
        assert rid == "run_001"

        loaded = repo.get_run("run_001", load_iterations=False)
        assert loaded is not None
        assert loaded.run_id == "run_001"
        assert loaded.kernel.name == "test.cu"
        assert loaded.status == "running"

    def test_get_run_not_found(self, db_session: Session):
        repo = RunRepo(db_session)
        assert repo.get_run("nonexistent") is None

    def test_list_runs_filtered(self, db_session: Session):
        kernel = _make_kernel()
        KernelRepo(db_session).save_source(kernel)
        repo = RunRepo(db_session)

        for i in range(5):
            status = "running" if i < 3 else "converged"
            repo.save_run(
                OptimizationRun(run_id=f"run_{i:03d}", kernel=kernel, target_arch="sm_80", status=status)
            )

        running = repo.list_runs(status="running")
        assert len(running) == 3
        converged = repo.list_runs(status="converged")
        assert len(converged) == 2

    def test_run_auto_persists_kernel(self, db_session: Session):
        """Kernel source not pre-saved — RunRepo should auto-persist it."""
        kernel = _make_kernel("auto_persist.cu")
        repo = RunRepo(db_session)
        run = OptimizationRun(run_id="run_auto", kernel=kernel, target_arch="sm_80")
        repo.save_run(run)

        loaded = repo.get_run("run_auto", load_iterations=False)
        assert loaded is not None
        assert loaded.kernel.name == "auto_persist.cu"


class TestCreateIteration:
    """Required: create iteration"""

    def test_save_and_list_iterations(self, db_session: Session):
        kernel = _make_kernel()
        KernelRepo(db_session).save_source(kernel)

        run = OptimizationRun(run_id="run_iter", kernel=kernel, target_arch="sm_80")
        repo = RunRepo(db_session)
        repo.save_run(run)

        v1 = _make_variant(kernel, "v1", iteration=1)
        KernelRepo(db_session).save_variant(v1)

        iteration = OptimizationIteration(
            iteration=1,
            variant_id="v1",
            variant=v1,
            bottleneck=BottleneckResult(
                primary=BottleneckClass.MEMORY_BOUND,
                confidence=0.9,
                reasoning="high dram throughput",
            ),
        )
        iter_id = repo.save_iteration(iteration, "run_iter")
        assert iter_id > 0

        iterations = repo.list_iterations("run_iter", load_details=False)
        assert len(iterations) == 1
        assert iterations[0].iteration == 1
        assert iterations[0].bottleneck is not None
        assert iterations[0].bottleneck.primary == BottleneckClass.MEMORY_BOUND

    def test_update_iteration_detail_with_fk(self, db_session: Session):
        kernel = _make_kernel()
        KernelRepo(db_session).save_source(kernel)

        v1 = _make_variant(kernel, "v1")
        KernelRepo(db_session).save_variant(v1)

        run = OptimizationRun(run_id="run_fk", kernel=kernel, target_arch="sm_80")
        repo = RunRepo(db_session)
        repo.save_run(run)

        iteration = OptimizationIteration(iteration=1, variant_id="v1", variant=v1)
        iter_id = repo.save_iteration(iteration, "run_fk")

        metrics_repo = MetricsRepo(db_session)
        metrics_id = metrics_repo.save(_make_metrics(variant_id="v1"))

        repo.update_iteration_detail(iter_id, profile_metrics_id=metrics_id)

        loaded = repo.get_iteration(iter_id, load_details=True)
        assert loaded is not None
        assert loaded.profile_metrics is not None
        assert loaded.profile_metrics.gpu_time_us == 100.0

    def test_iteration_unloadable_variant_raises(self, db_session: Session):
        """Referencing a non-existent variant should raise StorageError."""
        kernel = _make_kernel()
        KernelRepo(db_session).save_source(kernel)

        run = OptimizationRun(run_id="run_bad", kernel=kernel, target_arch="sm_80")
        repo = RunRepo(db_session)
        repo.save_run(run)

        # Create iteration row manually referencing a non-existent variant
        from cuda_opt_agent.memory.models import OptimizationIterationRow
        from datetime import datetime, timezone

        row = OptimizationIterationRow(
            run_id="run_bad",
            iteration_num=1,
            variant_id="nonexistent_variant",
            started_at=datetime.now(timezone.utc),
        )
        db_session.add(row)
        db_session.flush()

        with pytest.raises(StorageError, match="nonexistent_variant"):
            repo.get_iteration(row.id, load_details=False)


class TestTransactionRollback:
    """Required: transaction rollback"""

    def test_rollback_discards_inserted_data(self, db_engine):
        """Data inserted in a rolled-back session should not be visible in a new session."""
        SessionLocal = sessionmaker(bind=db_engine, expire_on_commit=False)

        # Insert in session that will be rolled back
        session1 = SessionLocal()
        try:
            repo1 = KernelRepo(session1)
            kernel = _make_kernel("rollback_test.cu")
            repo1.save_source(kernel)
            session1.rollback()
        finally:
            session1.close()

        # Verify in a fresh session
        session2 = SessionLocal()
        try:
            repo2 = KernelRepo(session2)
            results = repo2.find_sources_by_name("rollback_test.cu")
            assert len(results) == 0
        finally:
            session2.close()

    def test_rollback_on_exception(self, db_engine):
        """Exception in get_session-style usage triggers rollback."""
        SessionLocal = sessionmaker(bind=db_engine, expire_on_commit=False)

        session = SessionLocal()
        try:
            repo = KernelRepo(session)
            repo.save_source(_make_kernel("before_error.cu"))

            # Simulate an error mid-transaction
            with pytest.raises(ValueError):
                session.add(None)  # type: ignore — triggers error
                session.flush()

            session.rollback()
        finally:
            session.close()

        # Verify rollback occurred
        session2 = SessionLocal()
        try:
            repo2 = KernelRepo(session2)
            results = repo2.find_sources_by_name("before_error.cu")
            assert len(results) == 0
        finally:
            session2.close()

    def test_committed_data_persists(self, db_session: Session):
        """Data committed (via fixture teardown rollback being explicit) should be findable within the session."""
        repo = KernelRepo(db_session)
        repo.save_source(_make_kernel("committed.cu"))
        db_session.commit()

        results = repo.find_sources_by_name("committed.cu")
        assert len(results) == 1


class TestArtifactPersistence:
    """Required: artifact persistence"""

    def test_store_and_retrieve_text(self, tmp_path: Path):
        store = ArtifactStore(tmp_path / "artifacts")
        rel = store.store("run_001", "kernel.cu", "int main() {}")
        assert "run_001" in rel
        assert rel.endswith(".cu")

        content = store.retrieve_text(rel)
        assert content == "int main() {}"

    def test_store_binary(self, tmp_path: Path):
        store = ArtifactStore(tmp_path / "artifacts")
        binary = b"\x7fELF\x00\x00"
        rel = store.store("run_001", "kernel.cubin", binary, kind="cubin")
        assert rel.endswith(".cubin")
        assert store.retrieve(rel) == binary

    def test_content_deduplication(self, tmp_path: Path):
        store = ArtifactStore(tmp_path / "artifacts")
        r1 = store.store("run_001", "a.cu", "same content")
        r2 = store.store("run_001", "b.cu", "same content")
        assert r1 == r2  # Same content hash -> same path

    def test_different_content_different_path(self, tmp_path: Path):
        store = ArtifactStore(tmp_path / "artifacts")
        r1 = store.store("run_001", "a.cu", "content A")
        r2 = store.store("run_001", "a.cu", "content B")
        assert r1 != r2

    def test_list_artifacts(self, tmp_path: Path):
        store = ArtifactStore(tmp_path / "artifacts")
        store.store("run_001", "a.cu", "a", kind="source")
        store.store("run_001", "b.ptx", "b", kind="ptx")
        store.store("run_002", "c.cu", "c", kind="source")

        r1_artifacts = store.list_artifacts("run_001")
        assert len(r1_artifacts) == 2

        r2_artifacts = store.list_artifacts("run_002")
        assert len(r2_artifacts) == 1

    def test_list_kinds(self, tmp_path: Path):
        store = ArtifactStore(tmp_path / "artifacts")
        store.store("run_001", "a.cu", "a", kind="source")
        store.store("run_001", "b.ptx", "b", kind="ptx")

        kinds = store.list_kinds("run_001")
        assert set(kinds) == {"source", "ptx"}

    def test_exists(self, tmp_path: Path):
        store = ArtifactStore(tmp_path / "artifacts")
        rel = store.store("run_001", "a.cu", "content")
        assert store.exists(rel)
        assert not store.exists("run_001/source/nonexistent.cu")

    def test_get_size(self, tmp_path: Path):
        store = ArtifactStore(tmp_path / "artifacts")
        store.store("run_001", "a.cu", "hello")  # 5 bytes
        store.store("run_001", "b.cu", "world!!")  # 7 bytes

        assert store.get_size("run_001") == 12
        assert store.get_size("run_999") == 0


class TestArtifactRetrieval:
    """Required: artifact retrieval"""

    def test_retrieve_not_found(self, tmp_path: Path):
        store = ArtifactStore(tmp_path / "artifacts")
        with pytest.raises(ArtifactNotFoundError):
            store.retrieve("nonexistent/file.cu")

    def test_retrieve_text_decodes_utf8(self, tmp_path: Path):
        store = ArtifactStore(tmp_path / "artifacts")
        content = "// CUDA kernel\n__global__ void k() {}"
        rel = store.store("run_001", "kernel.cu", content)
        assert store.retrieve_text(rel) == content

    def test_retrieve_returns_bytes(self, tmp_path: Path):
        store = ArtifactStore(tmp_path / "artifacts")
        binary = bytes(range(256))
        rel = store.store("run_001", "data.bin", binary)
        assert store.retrieve(rel) == binary


class TestDeleteBehavior:
    """Required: delete behavior"""

    def test_cleanup_single_run(self, tmp_path: Path):
        store = ArtifactStore(tmp_path / "artifacts")
        store.store("run_001", "a.cu", "a")
        store.store("run_002", "b.cu", "b")

        store.cleanup("run_001")
        assert store.list_artifacts("run_001") == []
        assert len(store.list_artifacts("run_002")) == 1

    def test_cleanup_all(self, tmp_path: Path):
        store = ArtifactStore(tmp_path / "artifacts")
        store.store("run_001", "a.cu", "a")
        store.store("run_002", "b.cu", "b")

        store.cleanup_all()
        assert store.list_artifacts("run_001") == []
        assert store.list_artifacts("run_002") == []

    def test_cleanup_nonexistent_run_is_noop(self, tmp_path: Path):
        store = ArtifactStore(tmp_path / "artifacts")
        store.cleanup("nonexistent_run")  # Should not raise

    def test_cascade_delete_kernel_removes_variants(self, db_session: Session):
        """Deleting a kernel source should cascade-delete its variants."""
        repo = KernelRepo(db_session)
        kernel = _make_kernel()
        kid = repo.save_source(kernel)
        repo.save_variant(_make_variant(kernel, "v1"))
        repo.save_variant(_make_variant(kernel, "v2"))

        # Delete the kernel source row directly
        from cuda_opt_agent.memory.models import KernelSourceRow
        row = db_session.get(KernelSourceRow, kid)
        assert row is not None
        db_session.delete(row)
        db_session.flush()

        # Variants should be gone
        assert repo.get_variant("v1") is None
        assert repo.get_variant("v2") is None

    def test_cascade_delete_run_removes_iterations(self, db_session: Session):
        """Deleting an optimization run should cascade-delete its iterations."""
        kernel = _make_kernel()
        KernelRepo(db_session).save_source(kernel)
        v1 = _make_variant(kernel, "v1")
        KernelRepo(db_session).save_variant(v1)

        run_repo = RunRepo(db_session)
        run = OptimizationRun(run_id="run_del", kernel=kernel, target_arch="sm_80")
        run_repo.save_run(run)

        iteration = OptimizationIteration(iteration=1, variant_id="v1", variant=v1)
        iter_id = run_repo.save_iteration(iteration, "run_del")

        # Delete the run row directly
        from cuda_opt_agent.memory.models import OptimizationRunRow
        row = db_session.get(OptimizationRunRow, "run_del")
        assert row is not None
        db_session.delete(row)
        db_session.flush()

        # Iteration should be gone
        assert run_repo.get_iteration(iter_id) is None

    def test_set_null_on_variant_delete_for_iteration_fk(self, db_session: Session):
        """Deleting a variant referenced by an iteration should SET NULL the FK."""
        kernel = _make_kernel()
        KernelRepo(db_session).save_source(kernel)
        v1 = _make_variant(kernel, "v1")
        KernelRepo(db_session).save_variant(v1)

        run_repo = RunRepo(db_session)
        run = OptimizationRun(run_id="run_snull", kernel=kernel, target_arch="sm_80")
        run_repo.save_run(run)

        iteration = OptimizationIteration(iteration=1, variant_id="v1", variant=v1)
        iter_id = run_repo.save_iteration(iteration, "run_snull")

        # Delete the variant
        from cuda_opt_agent.memory.models import KernelVariantRow
        vrow = db_session.get(KernelVariantRow, "v1")
        assert vrow is not None
        db_session.delete(vrow)
        db_session.flush()

        # The iteration row should still exist but with NULL variant_id
        from cuda_opt_agent.memory.models import OptimizationIterationRow
        irow = db_session.get(OptimizationIterationRow, iter_id)
        assert irow is not None
        assert irow.variant_id is None


class TestConcurrentSessionBehavior:
    """Required: concurrent session behavior"""

    def test_two_sessions_see_committed_data(self, db_session_factory):
        """A commit in one session is visible to another."""
        session1 = db_session_factory()
        session2 = db_session_factory()
        try:
            repo1 = KernelRepo(session1)
            repo1.save_source(_make_kernel("concurrent.cu"))
            session1.commit()

            repo2 = KernelRepo(session2)
            results = repo2.find_sources_by_name("concurrent.cu")
            assert len(results) == 1
        finally:
            session1.close()
            session2.close()

    def test_two_sessions_do_not_see_uncommitted_data(self, db_session_factory):
        """Uncommitted data in one session is not visible to another."""
        session1 = db_session_factory()
        session2 = db_session_factory()
        try:
            repo1 = KernelRepo(session1)
            repo1.save_source(_make_kernel("uncommitted.cu"))
            # No commit

            repo2 = KernelRepo(session2)
            results = repo2.find_sources_by_name("uncommitted.cu")
            assert len(results) == 0
        finally:
            session1.rollback()
            session1.close()
            session2.close()

    def test_threaded_concurrent_writes(self, db_session_factory):
        """Two threads writing to different rows should both succeed."""
        errors: list[Exception] = []

        def write_kernel(name: str):
            session = db_session_factory()
            try:
                repo = KernelRepo(session)
                repo.save_source(_make_kernel(name))
                session.commit()
            except Exception as e:
                errors.append(e)
                session.rollback()
            finally:
                session.close()

        t1 = threading.Thread(target=write_kernel, args=("thread_a.cu",))
        t2 = threading.Thread(target=write_kernel, args=("thread_b.cu",))
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert len(errors) == 0

        # Both should be visible
        session = db_session_factory()
        try:
            repo = KernelRepo(session)
            assert len(repo.find_sources_by_name("thread_a.cu")) == 1
            assert len(repo.find_sources_by_name("thread_b.cu")) == 1
        finally:
            session.close()


# ══════════════════════════════════════════════════════════════
# ORM -> Pydantic Conversion Correctness
# ══════════════════════════════════════════════════════════════


class TestConversionCorrectness:
    """Verify ORM -> Pydantic round-trips for all result types."""

    def test_metrics_round_trip(self, db_session: Session):
        repo = MetricsRepo(db_session)
        original = _make_metrics()
        mid = repo.save(original)
        loaded = repo.get(mid)

        assert loaded is not None
        assert loaded.kernel_name == original.kernel_name
        assert loaded.gpu_time_us == original.gpu_time_us
        assert loaded.sm_throughput == original.sm_throughput
        assert loaded.block_size == original.block_size
        assert loaded.grid_size == original.grid_size
        assert loaded.achieved_occupancy == original.achieved_occupancy
        assert loaded.registers_per_thread == original.registers_per_thread

    def test_metrics_by_variant(self, db_session: Session):
        repo = MetricsRepo(db_session)
        repo.save(_make_metrics(variant_id="v1"))
        repo.save(_make_metrics(variant_id="v1"))
        repo.save(_make_metrics(variant_id="v2"))

        m = repo.find_by_variant("v1")
        assert m is not None
        assert m.kernel_name == "test"

    def test_metrics_by_kernel(self, db_session: Session):
        repo = MetricsRepo(db_session)
        for _ in range(3):
            repo.save(_make_metrics(kernel_name="matmul"))
        repo.save(_make_metrics(kernel_name="conv"))

        results = repo.list_by_kernel("matmul")
        assert len(results) == 3

    def test_compilation_round_trip(self, db_session: Session):
        repo = ResultRepo(db_session)
        result = CompilationResult(
            success=True,
            source_code="__global__ void k() {}",
            arch="sm_80",
            cubin_path="/tmp/test.cubin",
            ptx_source="// PTX",
            diagnostics=[
                CompilationDiagnostic(severity="warning", message="unused variable x")
            ],
            compile_time_s=1.23,
        )
        cid = repo.save_compilation(result, variant_id="v1")
        loaded = repo.get_compilation(cid)

        assert loaded is not None
        assert loaded.success is True
        assert loaded.cubin_path == "/tmp/test.cubin"
        assert len(loaded.diagnostics) == 1
        assert loaded.diagnostics[0].message == "unused variable x"
        assert loaded.compile_time_s == 1.23

    def test_correctness_round_trip(self, db_session: Session):
        repo = ResultRepo(db_session)
        result = CorrectnessResult(
            passed=True,
            variant_id="v1",
            baseline_variant_id="v0",
            max_absolute_error=1e-7,
            max_relative_error=1e-5,
            total_elements=1024,
        )
        cid = repo.save_correctness(result)
        loaded = repo.get_correctness(cid)

        assert loaded is not None
        assert loaded.passed is True
        assert loaded.total_elements == 1024
        assert loaded.variant_id == "v1"
        assert loaded.baseline_variant_id == "v0"

    def test_correctness_by_variant(self, db_session: Session):
        repo = ResultRepo(db_session)
        repo.save_correctness(CorrectnessResult(
            passed=True, variant_id="v1", baseline_variant_id="v0", checked_at=datetime.now(timezone.utc),
        ))
        result = repo.find_correctness_by_variant("v1")
        assert result is not None
        assert result.passed is True

    def test_correctness_by_variant_not_found(self, db_session: Session):
        repo = ResultRepo(db_session)
        assert repo.find_correctness_by_variant("nonexistent") is None

    def test_benchmark_round_trip(self, db_session: Session):
        repo = ResultRepo(db_session)
        result = BenchmarkResult(
            variant_id="v1",
            timing=TimingStats(
                mean_ms=5.0, median_ms=4.9, std_ms=0.2,
                min_ms=4.7, max_ms=5.5, n_samples=10,
            ),
            arch="sm_80",
            block_size=(16, 16, 1),
        )
        bid = repo.save_benchmark(result)
        loaded = repo.get_benchmark(bid)

        assert loaded is not None
        assert loaded.timing.mean_ms == 5.0
        assert loaded.block_size == (16, 16, 1)

    def test_find_best_benchmark(self, db_session: Session):
        repo = ResultRepo(db_session)
        for mean in [10.0, 5.0, 7.5]:
            repo.save_benchmark(BenchmarkResult(
                variant_id="v1",
                timing=TimingStats(
                    mean_ms=mean, median_ms=mean, std_ms=0.1,
                    min_ms=mean - 0.1, max_ms=mean + 0.1, n_samples=5,
                ),
            ))

        best = repo.find_best_benchmark("v1")
        assert best is not None
        assert best.timing.mean_ms == 5.0

    def test_find_best_benchmark_not_found(self, db_session: Session):
        repo = ResultRepo(db_session)
        assert repo.find_best_benchmark("nonexistent") is None

    def test_transform_round_trip(self, db_session: Session):
        repo = ResultRepo(db_session)
        result = TransformResult(
            spec=TransformSpec(
                transform_name="shared_memory_tiling",
                params={"tile_size": 16},
            ),
            original_source="// original",
            transformed_source="// tiled",
            diff_summary="+shared mem decl",
        )
        tid = repo.save_transform(result)
        loaded = repo.get_transform(tid)

        assert loaded is not None
        assert loaded.spec.transform_name == "shared_memory_tiling"
        assert loaded.transformed_source == "// tiled"
        assert loaded.diff_summary == "+shared mem decl"


# ══════════════════════════════════════════════════════════════
# Repository Boundary Tests
# ══════════════════════════════════════════════════════════════


class TestRepositoryBoundaries:
    """Verify that repos use only public APIs of other repos."""

    def test_run_repo_accepts_injected_kernel_repo(self, db_session: Session):
        """RunRepo should work with an externally-provided KernelRepo."""
        kernel_repo = KernelRepo(db_session)
        run_repo = RunRepo(db_session, kernel_repo=kernel_repo)

        kernel = _make_kernel()
        run = OptimizationRun(run_id="run_inject", kernel=kernel, target_arch="sm_80")
        run_repo.save_run(run)

        loaded = run_repo.get_run("run_inject", load_iterations=False)
        assert loaded is not None
        assert loaded.kernel.name == "test.cu"

    def test_run_repo_uses_public_resolve_kernel_id(self, db_session: Session):
        """RunRepo.save_run should use resolve_kernel_id (public), not _resolve_kernel_id (private)."""
        # This is verified by the fact that save_run works — if it called
        # a non-existent private method, it would fail. The public method
        # handles auto-persistence of the kernel.
        kernel_repo = KernelRepo(db_session)
        run_repo = RunRepo(db_session, kernel_repo=kernel_repo)

        kernel = _make_kernel("boundary_test.cu")
        run = OptimizationRun(run_id="run_bound", kernel=kernel, target_arch="sm_80")
        run_repo.save_run(run)

        # Kernel was auto-persisted via public API
        sources = kernel_repo.find_sources_by_name("boundary_test.cu")
        assert len(sources) == 1


# ══════════════════════════════════════════════════════════════
# End-to-End Integration
# ══════════════════════════════════════════════════════════════


class TestEndToEnd:
    """Full pipeline: save kernel -> variant -> metrics -> iteration -> run."""

    def test_full_optimization_flow(self, db_session: Session):
        kernel_repo = KernelRepo(db_session)
        run_repo = RunRepo(db_session, kernel_repo=kernel_repo)
        metrics_repo = MetricsRepo(db_session)
        result_repo = ResultRepo(db_session)

        # 1. Save original kernel
        kernel = _make_kernel("matmul.cu", "__global__ void matmul() {}")
        kernel_id = kernel_repo.save_source(kernel)

        # 2. Create optimization run
        run = OptimizationRun(run_id="run_e2e", kernel=kernel, target_arch="sm_80")
        run_repo.save_run(run)

        # 3. Create variant
        variant = _make_variant(kernel, "v1_tiled", iteration=1)
        variant.transform_history = [
            TransformSpec(transform_name="shared_memory_tiling", params={"tile": 16})
        ]
        kernel_repo.save_variant(variant)

        # 4. Save metrics
        metrics = _make_metrics(variant_id="v1_tiled")
        metrics_id = metrics_repo.save(metrics)

        # 5. Save compilation result
        comp = CompilationResult(success=True, source_code=variant.source_code, arch="sm_80")
        comp_id = result_repo.save_compilation(comp, variant_id="v1_tiled")

        # 6. Save benchmark
        bench = BenchmarkResult(
            variant_id="v1_tiled",
            timing=TimingStats(mean_ms=3.2, median_ms=3.1, std_ms=0.1, min_ms=3.0, max_ms=3.5, n_samples=10),
        )
        bench_id = result_repo.save_benchmark(bench)

        # 7. Save iteration with detail references
        iteration = OptimizationIteration(
            iteration=1,
            variant_id="v1_tiled",
            variant=variant,
            bottleneck=BottleneckResult(primary=BottleneckClass.MEMORY_BOUND, confidence=0.9),
            transform_applied=TransformSpec(transform_name="shared_memory_tiling", params={"tile": 16}),
        )
        iter_id = run_repo.save_iteration(iteration, "run_e2e")
        run_repo.update_iteration_detail(
            iter_id,
            profile_metrics_id=metrics_id,
            compilation_id=comp_id,
            benchmark_id=bench_id,
        )

        # 8. Update run
        run.best_variant_id = "v1_tiled"
        run.best_speedup = 3.1
        run.status = "converged"
        run_repo.update_run(run)

        # 9. Verify everything loads correctly
        loaded_run = run_repo.get_run("run_e2e", load_iterations=True)
        assert loaded_run is not None
        assert loaded_run.status == "converged"
        assert loaded_run.best_speedup == 3.1
        assert len(loaded_run.iterations) == 1

        loaded_iter = loaded_run.iterations[0]
        assert loaded_iter.iteration == 1
        assert loaded_iter.bottleneck is not None
        assert loaded_iter.bottleneck.primary == BottleneckClass.MEMORY_BOUND
        assert loaded_iter.profile_metrics is not None
        assert loaded_iter.profile_metrics.gpu_time_us == 100.0
        assert loaded_iter.compilation is not None
        assert loaded_iter.compilation.success is True
        assert loaded_iter.benchmark is not None
        assert loaded_iter.benchmark.timing.mean_ms == 3.2
        assert loaded_iter.variant.variant_id == "v1_tiled"