"""Repository for Compilation, Correctness, Benchmark, and Transform results."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from cuda_opt_agent.memory.models import (
    BenchmarkResultRow,
    CompilationResultRow,
    CorrectnessResultRow,
    TransformResultRow,
)
from cuda_opt_agent.memory.repositories.base import (
    BaseRepository,
    from_json_dict,
    from_json_list,
    to_json_dict,
    to_json_list,
)
from cuda_opt_agent.schemas.benchmark import BenchmarkResult, SpeedupResult, TimingStats
from cuda_opt_agent.schemas.compilation import CompilationDiagnostic, CompilationResult
from cuda_opt_agent.schemas.correctness import CorrectnessResult
from cuda_opt_agent.schemas.transform import TransformResult, TransformSpec


class ResultRepo:
    """CRUD for all pipeline result types."""

    def __init__(self, session: Session) -> None:
        self._session = session
        self._comp_repo = BaseRepository(CompilationResultRow, session)
        self._corr_repo = BaseRepository(CorrectnessResultRow, session)
        self._bench_repo = BaseRepository(BenchmarkResultRow, session)
        self._trans_repo = BaseRepository(TransformResultRow, session)

    # ── Compilation Results ────────────────────────────────────

    def save_compilation(self, result: CompilationResult, variant_id: str | None = None) -> int:
        """Persist a CompilationResult. Returns the auto-generated DB id."""
        row = CompilationResultRow(
            variant_id=variant_id,
            success=result.success,
            source_code=result.source_code,
            arch=result.arch,
            cubin_path=result.cubin_path,
            ptx_path=result.ptx_path,
            ptx_source=result.ptx_source,
            diagnostics_json=[d.model_dump(mode="json") for d in result.diagnostics] or None,
            compile_time_s=result.compile_time_s,
            compiled_at=result.compiled_at,
            stdout=result.stdout,
            stderr=result.stderr,
            returncode=result.returncode,
        )
        self._comp_repo.add(row)
        return row.id  # type: ignore[return-value]

    def get_compilation(self, id: int) -> CompilationResult | None:
        row = self._comp_repo.get_by_id(id)
        if row is None:
            return None
        return self._compilation_to_pydantic(row)

    # ── Correctness Results ────────────────────────────────────

    def save_correctness(self, result: CorrectnessResult) -> int:
        row = CorrectnessResultRow(
            variant_id=result.variant_id,
            baseline_variant_id=result.baseline_variant_id,
            passed=result.passed,
            max_absolute_error=result.max_absolute_error,
            max_relative_error=result.max_relative_error,
            tolerance_absolute=result.tolerance_absolute,
            tolerance_relative=result.tolerance_relative,
            failed_elements=result.failed_elements,
            total_elements=result.total_elements,
            message=result.message,
            checked_at=result.checked_at,
            metadata_json=result.metadata or None,
        )
        self._corr_repo.add(row)
        return row.id  # type: ignore[return-value]

    def get_correctness(self, id: int) -> CorrectnessResult | None:
        row = self._corr_repo.get_by_id(id)
        if row is None:
            return None
        return self._correctness_to_pydantic(row)

    def find_correctness_by_variant(self, variant_id: str) -> CorrectnessResult | None:
        """Most recent correctness result for a variant."""
        stmt = (
            select(CorrectnessResultRow)
            .where(CorrectnessResultRow.variant_id == variant_id)
            .order_by(CorrectnessResultRow.checked_at.desc())
            .limit(1)
        )
        row = self._session.execute(stmt).scalars().first()
        if row is None:
            return None
        return self._correctness_to_pydantic(row)

    # ── Benchmark Results ──────────────────────────────────────

    def save_benchmark(self, result: BenchmarkResult) -> int:
        row = BenchmarkResultRow(
            variant_id=result.variant_id,
            timing_json=result.timing.model_dump(mode="json"),
            arch=result.arch,
            block_size_json=list(result.block_size) if result.block_size else None,
            grid_size_json=list(result.grid_size) if result.grid_size else None,
            correct=result.correct,
            benchmarked_at=result.benchmarked_at,
            metadata_json=result.metadata or None,
        )
        self._bench_repo.add(row)
        return row.id  # type: ignore[return-value]

    def get_benchmark(self, id: int) -> BenchmarkResult | None:
        row = self._bench_repo.get_by_id(id)
        if row is None:
            return None
        return self._benchmark_to_pydantic(row)

    def find_benchmarks_by_variant(self, variant_id: str) -> list[BenchmarkResult]:
        """All benchmark runs for a variant."""
        stmt = (
            select(BenchmarkResultRow)
            .where(BenchmarkResultRow.variant_id == variant_id)
            .order_by(BenchmarkResultRow.benchmarked_at.desc())
        )
        rows = self._session.execute(stmt).scalars().all()
        return [self._benchmark_to_pydantic(r) for r in rows]

    def find_best_benchmark(self, variant_id: str) -> BenchmarkResult | None:
        """Fastest benchmark for a variant (lowest mean time)."""
        stmt = (
            select(BenchmarkResultRow)
            .where(BenchmarkResultRow.variant_id == variant_id)
            .order_by(BenchmarkResultRow.benchmarked_at.desc())
        )
        rows = self._session.execute(stmt).scalars().all()
        if not rows:
            return None
        return min(
            (self._benchmark_to_pydantic(r) for r in rows),
            key=lambda b: b.timing.mean_ms,
        )

    # ── Transform Results ──────────────────────────────────────

    def save_transform(self, result: TransformResult) -> int:
        row = TransformResultRow(
            spec_json=result.spec.model_dump(mode="json"),
            original_source=result.original_source,
            transformed_source=result.transformed_source,
            success=result.success,
            error_message=result.error_message,
            warnings_json=result.warnings or None,
            diff_summary=result.diff_summary,
            applied_at=result.applied_at,
            metadata_json=result.metadata or None,
        )
        self._trans_repo.add(row)
        return row.id  # type: ignore[return-value]

    def get_transform(self, id: int) -> TransformResult | None:
        row = self._trans_repo.get_by_id(id)
        if row is None:
            return None
        return self._transform_to_pydantic(row)

    # ── Conversion helpers ─────────────────────────────────────

    @staticmethod
    def _compilation_to_pydantic(row: CompilationResultRow) -> CompilationResult:
        diagnostics = [
            CompilationDiagnostic.model_validate(d)
            for d in (row.diagnostics_json or [])
        ]
        return CompilationResult(
            success=row.success,
            source_code=row.source_code,
            arch=row.arch,
            cubin_path=row.cubin_path,
            ptx_path=row.ptx_path,
            ptx_source=row.ptx_source,
            diagnostics=diagnostics,
            compile_time_s=row.compile_time_s,
            compiled_at=row.compiled_at,
            stdout=row.stdout,
            stderr=row.stderr,
            returncode=row.returncode,
        )

    @staticmethod
    def _correctness_to_pydantic(row: CorrectnessResultRow) -> CorrectnessResult:
        return CorrectnessResult(
            passed=row.passed,
            variant_id=row.variant_id,
            baseline_variant_id=row.baseline_variant_id,
            max_absolute_error=row.max_absolute_error,
            max_relative_error=row.max_relative_error,
            tolerance_absolute=row.tolerance_absolute,
            tolerance_relative=row.tolerance_relative,
            failed_elements=row.failed_elements,
            total_elements=row.total_elements,
            message=row.message,
            checked_at=row.checked_at,
            metadata=row.metadata_json or {},
        )

    @staticmethod
    def _benchmark_to_pydantic(row: BenchmarkResultRow) -> BenchmarkResult:
        return BenchmarkResult(
            variant_id=row.variant_id,
            timing=TimingStats.model_validate(row.timing_json),
            arch=row.arch,
            block_size=tuple(row.block_size_json) if row.block_size_json else None,
            grid_size=tuple(row.grid_size_json) if row.grid_size_json else None,
            correct=row.correct,
            benchmarked_at=row.benchmarked_at,
            metadata=row.metadata_json or {},
        )

    @staticmethod
    def _transform_to_pydantic(row: TransformResultRow) -> TransformResult:
        return TransformResult(
            spec=TransformSpec.model_validate(row.spec_json),
            original_source=row.original_source,
            transformed_source=row.transformed_source,
            success=row.success,
            error_message=row.error_message,
            warnings=row.warnings_json or [],
            diff_summary=row.diff_summary,
            applied_at=row.applied_at,
            metadata=row.metadata_json or {},
        )