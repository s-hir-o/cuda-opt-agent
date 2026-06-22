"""Repository for OptimizationRun and OptimizationIteration persistence."""

from __future__ import annotations

from datetime import datetime
from typing import Sequence

from sqlalchemy import select
from sqlalchemy.orm import Session

from cuda_opt_agent.core.exceptions import StorageError
from cuda_opt_agent.memory.models import (
    KernelSourceRow,
    OptimizationIterationRow,
    OptimizationRunRow,
)
from cuda_opt_agent.memory.repositories.base import (
    BaseRepository,
    from_json_dict,
    to_json_dict,
)
from cuda_opt_agent.memory.repositories.kernel_repo import KernelRepo
from cuda_opt_agent.schemas.bottleneck import BottleneckResult
from cuda_opt_agent.schemas.cost_model import CostEstimate
from cuda_opt_agent.schemas.features import KernelFeatureVector
from cuda_opt_agent.schemas.kernel import KernelSource, KernelVariant
from cuda_opt_agent.schemas.optimization import OptimizationIteration, OptimizationRun
from cuda_opt_agent.schemas.transform import TransformSpec


class RunRepo:
    """CRUD for optimization runs and their iterations.

    Depends on KernelRepo for kernel/variant resolution.
    Caller must provide a KernelRepo that shares the same session.
    """

    def __init__(self, session: Session, kernel_repo: KernelRepo | None = None) -> None:
        self._session = session
        self._run_repo = BaseRepository(OptimizationRunRow, session)
        self._iter_repo = BaseRepository(OptimizationIterationRow, session)
        self._kernel_repo = kernel_repo or KernelRepo(session)

    # ── Optimization Run ───────────────────────────────────────

    def save_run(self, run: OptimizationRun) -> str:
        """Persist an OptimizationRun header. Returns run_id.

        Does NOT persist iterations — use save_iteration() for each.
        """
        kernel_id = self._kernel_repo.resolve_kernel_id(run.kernel)
        row = OptimizationRunRow(
            id=run.run_id,
            kernel_id=kernel_id,
            target_arch=run.target_arch,
            max_iterations=run.max_iterations,
            best_variant_id=run.best_variant_id,
            best_speedup=run.best_speedup,
            started_at=run.started_at,
            completed_at=run.completed_at,
            status=run.status,
            config_snapshot_json=run.config_snapshot or None,
            metadata_json=run.metadata or None,
        )
        self._run_repo.add(row)
        return row.id

    def get_run(
        self, run_id: str, load_iterations: bool = True
    ) -> OptimizationRun | None:
        """Load an OptimizationRun by run_id.

        Args:
            run_id: The run identifier.
            load_iterations: If True, eagerly loads all iterations.
        """
        row = self._run_repo.get_by_id(run_id)
        if row is None:
            return None

        kernel = self._kernel_repo.get_source(row.kernel_id)
        if kernel is None:
            return None

        run = self._run_to_pydantic(row, kernel)

        if load_iterations:
            run.iterations = self.list_iterations(run_id)

        return run

    def update_run(self, run: OptimizationRun) -> None:
        """Update mutable run fields (best_variant, status, completed_at, etc.)."""
        row = self._run_repo.get_by_id(run.run_id)
        if row is None:
            return
        row.best_variant_id = run.best_variant_id
        row.best_speedup = run.best_speedup
        row.status = run.status
        row.completed_at = run.completed_at
        row.metadata_json = run.metadata or None
        self._run_repo.update(row)

    def list_runs(
        self,
        kernel_id: int | None = None,
        status: str | None = None,
        offset: int = 0,
        limit: int = 100,
    ) -> list[OptimizationRun]:
        """List runs with optional filters. Does NOT load iterations."""
        stmt = select(OptimizationRunRow)
        if kernel_id is not None:
            stmt = stmt.where(OptimizationRunRow.kernel_id == kernel_id)
        if status is not None:
            stmt = stmt.where(OptimizationRunRow.status == status)
        stmt = stmt.order_by(OptimizationRunRow.started_at.desc()).offset(offset).limit(limit)

        rows = self._session.execute(stmt).scalars().all()
        results: list[OptimizationRun] = []
        for row in rows:
            kernel = self._kernel_repo.get_source(row.kernel_id)
            if kernel is not None:
                results.append(self._run_to_pydantic(row, kernel, include_iterations=False))
        return results

    # ── Optimization Iteration ─────────────────────────────────

    def save_iteration(self, iteration: OptimizationIteration, run_id: str) -> int:
        """Persist an OptimizationIteration. Returns the auto-generated DB id.

        Detail records (metrics, benchmark, etc.) must be saved separately first
        via their respective repos. Then set the FK id on the iteration row.
        """
        variant_id = iteration.variant.variant_id
        row = OptimizationIterationRow(
            run_id=run_id,
            iteration_num=iteration.iteration,
            variant_id=variant_id,
            started_at=iteration.started_at,
            completed_at=iteration.completed_at,
            error=iteration.error,
            # Inline JSON
            bottleneck_json=to_json_dict(iteration.bottleneck),
            transform_spec_json=to_json_dict(iteration.transform_applied),
            speedup_json=to_json_dict(iteration.speedup),
            analysis_json=to_json_dict(iteration.analysis) if iteration.analysis else None,
            features_json=to_json_dict(iteration.features) if iteration.features else None,
            cost_estimate_json=to_json_dict(iteration.cost_estimate) if iteration.cost_estimate else None,
        )
        self._iter_repo.add(row)
        return row.id

    def update_iteration_detail(
        self,
        iteration_db_id: int,
        *,
        profile_metrics_id: int | None = None,
        compilation_id: int | None = None,
        correctness_id: int | None = None,
        benchmark_id: int | None = None,
        transform_result_id: int | None = None,
        completed_at: datetime | None = None,
        error: str | None = None,
        **inline_json_updates: dict | None,
    ) -> None:
        """Update FK references and status fields on an iteration row.

        Called as pipeline stages complete.
        """
        row = self._iter_repo.get_by_id(iteration_db_id)
        if row is None:
            return

        if profile_metrics_id is not None:
            row.profile_metrics_id = profile_metrics_id
        if compilation_id is not None:
            row.compilation_id = compilation_id
        if correctness_id is not None:
            row.correctness_id = correctness_id
        if benchmark_id is not None:
            row.benchmark_id = benchmark_id
        if transform_result_id is not None:
            row.transform_result_id = transform_result_id
        if completed_at is not None:
            row.completed_at = completed_at
        if error is not None:
            row.error = error

        for field, value in inline_json_updates.items():
            if hasattr(row, field):
                setattr(row, field, value)

        self._iter_repo.update(row)

    def get_iteration(
        self, iteration_id: int, load_details: bool = True
    ) -> OptimizationIteration | None:
        """Load an OptimizationIteration by DB id."""
        row = self._iter_repo.get_by_id(iteration_id)
        if row is None:
            return None
        return self._iteration_to_pydantic(row, load_details=load_details)

    def list_iterations(
        self, run_id: str, load_details: bool = True
    ) -> list[OptimizationIteration]:
        """All iterations for a run, ordered by iteration number."""
        stmt = (
            select(OptimizationIterationRow)
            .where(OptimizationIterationRow.run_id == run_id)
            .order_by(OptimizationIterationRow.iteration_num)
        )
        rows = self._session.execute(stmt).scalars().all()
        return [
            self._iteration_to_pydantic(r, load_details=load_details) for r in rows
        ]

    # ── Conversion helpers ─────────────────────────────────────

    def _run_to_pydantic(
        self, row: OptimizationRunRow, kernel: KernelSource, include_iterations: bool = False
    ) -> OptimizationRun:
        run = OptimizationRun(
            run_id=row.id,
            kernel=kernel,
            target_arch=row.target_arch,
            max_iterations=row.max_iterations,
            best_variant_id=row.best_variant_id,
            best_speedup=row.best_speedup,
            started_at=row.started_at,
            completed_at=row.completed_at,
            status=row.status,
            config_snapshot=row.config_snapshot_json or {},
            metadata=row.metadata_json or {},
        )
        if include_iterations:
            run.iterations = self.list_iterations(row.id)
        return run

    def _iteration_to_pydantic(
        self, row: OptimizationIterationRow, load_details: bool = True
    ) -> OptimizationIteration:
        # Variant is required — load from kernel repo
        variant: KernelVariant | None = None
        if row.variant_id is not None:
            variant = self._kernel_repo.get_variant(row.variant_id)

        if variant is None:
            raise StorageError(
                f"Iteration {row.id} references variant '{row.variant_id}' "
                "which could not be loaded"
            )

        iteration = OptimizationIteration(
            iteration=row.iteration_num,
            variant_id=row.variant_id or "",
            variant=variant,
            started_at=row.started_at,
            completed_at=row.completed_at,
            error=row.error,
            # Inline JSON
            bottleneck=from_json_dict(row.bottleneck_json, BottleneckResult),
            transform_applied=from_json_dict(row.transform_spec_json, TransformSpec),
            speedup=from_json_dict(row.speedup_json, __import__(
                "cuda_opt_agent.schemas.benchmark", fromlist=["SpeedupResult"]
            ).SpeedupResult),
            analysis=row.analysis_json,           # kept as raw dict (Any)
            features=from_json_dict(row.features_json, KernelFeatureVector),
            cost_estimate=from_json_dict(row.cost_estimate_json, CostEstimate),
        )

        # Load detail records if requested
        if load_details:
            self._load_iteration_details(row, iteration)

        return iteration

    def _load_iteration_details(
        self, row: OptimizationIterationRow, iteration: OptimizationIteration
    ) -> None:
        """Eagerly load FK-referenced detail records."""
        if row.profile_metrics_id is not None:
            from cuda_opt_agent.memory.repositories.metrics_repo import MetricsRepo
            iteration.profile_metrics = MetricsRepo(self._session).get(row.profile_metrics_id)

        if row.compilation_id is not None:
            from cuda_opt_agent.memory.repositories.result_repo import ResultRepo
            result_repo = ResultRepo(self._session)
            iteration.compilation = result_repo.get_compilation(row.compilation_id)

        if row.correctness_id is not None:
            from cuda_opt_agent.memory.repositories.result_repo import ResultRepo
            result_repo = ResultRepo(self._session)
            iteration.correctness = result_repo.get_correctness(row.correctness_id)

        if row.benchmark_id is not None:
            from cuda_opt_agent.memory.repositories.result_repo import ResultRepo
            result_repo = ResultRepo(self._session)
            iteration.benchmark = result_repo.get_benchmark(row.benchmark_id)

        if row.transform_result_id is not None:
            from cuda_opt_agent.memory.repositories.result_repo import ResultRepo
            result_repo = ResultRepo(self._session)
            iteration.transform_result = result_repo.get_transform(row.transform_result_id)