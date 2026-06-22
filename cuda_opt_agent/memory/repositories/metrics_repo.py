"""Repository for ProfileMetrics persistence."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from cuda_opt_agent.memory.models import ProfileMetricsRow
from cuda_opt_agent.memory.repositories.base import BaseRepository
from cuda_opt_agent.schemas.metrics import ProfileMetrics


class MetricsRepo:
    """CRUD for ProfileMetrics records."""

    def __init__(self, session: Session) -> None:
        self._session = session
        self._repo = BaseRepository(ProfileMetricsRow, session)

    def save(self, metrics: ProfileMetrics) -> int:
        """Persist a ProfileMetrics. Returns the auto-generated DB id."""
        row = ProfileMetricsRow(
            variant_id=metrics.metadata.get("variant_id"),
            kernel_name=metrics.kernel_name,
            gpu_name=metrics.gpu_name,
            arch=metrics.arch,
            gpu_time_ns=metrics.gpu_time_ns,
            gpu_time_us=metrics.gpu_time_us,
            sm_throughput=metrics.sm_throughput,
            dram_throughput=metrics.dram_throughput,
            l1_throughput=metrics.l1_throughput,
            l2_throughput=metrics.l2_throughput,
            achieved_occupancy=metrics.achieved_occupancy,
            theoretical_occupancy=metrics.theoretical_occupancy,
            registers_per_thread=metrics.registers_per_thread,
            block_size_json=list(metrics.block_size) if metrics.block_size else None,
            grid_size_json=list(metrics.grid_size) if metrics.grid_size else None,
            shared_mem_bytes=metrics.shared_mem_bytes,
            global_load_efficiency=metrics.global_load_efficiency,
            global_store_efficiency=metrics.global_store_efficiency,
            l2_hit_rate=metrics.l2_hit_rate,
            l1_hit_rate=metrics.l1_hit_rate,
            occupancy_limited_by_registers=metrics.occupancy_limited_by_registers,
            occupancy_limited_by_shared_mem=metrics.occupancy_limited_by_shared_mem,
            occupancy_limited_by_block_size=metrics.occupancy_limited_by_block_size,
            raw_metrics_json=metrics.raw_metrics or None,
            profiled_at=metrics.profiled_at,
        )
        self._repo.add(row)
        return row.id  # type: ignore[return-value]

    def get(self, id: int) -> ProfileMetrics | None:
        """Load a ProfileMetrics by DB id."""
        row = self._repo.get_by_id(id)
        if row is None:
            return None
        return self._to_pydantic(row)

    def find_by_variant(self, variant_id: str) -> ProfileMetrics | None:
        """Most recent metrics for a variant."""
        stmt = (
            select(ProfileMetricsRow)
            .where(ProfileMetricsRow.variant_id == variant_id)
            .order_by(ProfileMetricsRow.profiled_at.desc())
            .limit(1)
        )
        row = self._session.execute(stmt).scalars().first()
        if row is None:
            return None
        return self._to_pydantic(row)

    def list_by_kernel(
        self, kernel_name: str, offset: int = 0, limit: int = 50
    ) -> list[ProfileMetrics]:
        """All metrics for a kernel name, most recent first."""
        stmt = (
            select(ProfileMetricsRow)
            .where(ProfileMetricsRow.kernel_name == kernel_name)
            .order_by(ProfileMetricsRow.profiled_at.desc())
            .offset(offset)
            .limit(limit)
        )
        rows = self._session.execute(stmt).scalars().all()
        return [self._to_pydantic(r) for r in rows]

    # ── Conversion ─────────────────────────────────────────────

    @staticmethod
    def _to_pydantic(row: ProfileMetricsRow) -> ProfileMetrics:
        return ProfileMetrics(
            kernel_name=row.kernel_name,
            gpu_name=row.gpu_name,
            arch=row.arch,
            gpu_time_ns=row.gpu_time_ns,
            gpu_time_us=row.gpu_time_us,
            sm_throughput=row.sm_throughput,
            dram_throughput=row.dram_throughput,
            l1_throughput=row.l1_throughput,
            l2_throughput=row.l2_throughput,
            achieved_occupancy=row.achieved_occupancy,
            theoretical_occupancy=row.theoretical_occupancy,
            registers_per_thread=row.registers_per_thread,
            block_size=tuple(row.block_size_json) if row.block_size_json else None,
            grid_size=tuple(row.grid_size_json) if row.grid_size_json else None,
            shared_mem_bytes=rows.shared_mem_bytes,
            global_load_efficiency=row.global_load_efficiency,
            global_store_efficiency=row.global_store_efficiency,
            l2_hit_rate=row.l2_hit_rate,
            l1_hit_rate=row.l1_hit_rate,
            occupancy_limited_by_registers=row.occupancy_limited_by_registers,
            occupancy_limited_by_shared_mem=row.occupancy_limited_by_shared_mem,
            occupancy_limited_by_block_size=row.occupancy_limited_by_block_size,
            raw_metrics=row.raw_metrics_json or {},
            profiled_at=row.profiled_at,
        )