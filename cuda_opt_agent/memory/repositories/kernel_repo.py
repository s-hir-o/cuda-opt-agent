"""Repository for KernelSource and KernelVariant persistence."""

from __future__ import annotations

from datetime import datetime
from typing import Sequence

from sqlalchemy import select
from sqlalchemy.orm import Session

from cuda_opt_agent.core.exceptions import StorageError
from cuda_opt_agent.memory.models import KernelSourceRow, KernelVariantRow
from cuda_opt_agent.memory.repositories.base import (
    BaseRepository,
    to_json_dict,
    to_json_list,
    from_json_list,
)
from cuda_opt_agent.schemas.kernel import KernelLanguage, KernelSource, KernelVariant
from cuda_opt_agent.schemas.transform import TransformSpec


class KernelRepo:
    """CRUD for kernel sources and their variants."""

    def __init__(self, session: Session) -> None:
        self._session = session
        self._src_repo = BaseRepository(KernelSourceRow, session)
        self._var_repo = BaseRepository(KernelVariantRow, session)

    # ── Kernel Source ──────────────────────────────────────────

    def save_source(self, kernel: KernelSource) -> int:
        """Persist a KernelSource. Returns the auto-generated DB id."""
        row = KernelSourceRow(
            name=kernel.name,
            language=kernel.language.value,
            source_code=kernel.source_code,
            file_path=kernel.file_path,
            entry_point=kernel.entry_point,
            submitted_at=kernel.submitted_at,
            metadata_json=kernel.metadata or None,
        )
        self._src_repo.add(row)
        return row.id  # type: ignore[return-value]

    def get_source(self, id: int) -> KernelSource | None:
        """Load a KernelSource by DB id."""
        row = self._src_repo.get_by_id(id)
        if row is None:
            return None
        return self._source_to_pydantic(row)

    def update_source(self, kernel: KernelSource, db_id: int) -> None:
        """Update mutable fields on an existing KernelSource row."""
        row = self._src_repo.get_by_id(db_id)
        if row is None:
            return
        row.name = kernel.name
        row.language = kernel.language.value
        row.source_code = kernel.source_code
        row.file_path = kernel.file_path
        row.entry_point = kernel.entry_point
        row.metadata_json = kernel.metadata or None
        self._src_repo.update(row)

    def find_sources_by_name(self, name: str) -> list[KernelSource]:
        """Find kernel sources by exact name."""
        stmt = select(KernelSourceRow).where(KernelSourceRow.name == name)
        rows = self._session.execute(stmt).scalars().all()
        return [self._source_to_pydantic(r) for r in rows]

    def list_sources(self, offset: int = 0, limit: int = 100) -> list[KernelSource]:
        """Paginated list of all kernel sources."""
        rows = self._src_repo.list_all(offset=offset, limit=limit)
        return [self._source_to_pydantic(r) for r in rows]

    # ── Kernel Variant ─────────────────────────────────────────

    def save_variant(self, variant: KernelVariant) -> str:
        """Persist a KernelVariant. Returns the variant_id."""
        row = KernelVariantRow(
            id=variant.variant_id,
            parent_kernel_id=self.resolve_kernel_id(variant.parent_source),
            source_code=variant.source_code,
            language=variant.language.value,
            entry_point=variant.entry_point,
            transform_history_json=to_json_list(variant.transform_history) or None,
            iteration=variant.iteration,
            created_at=variant.created_at,
            metadata_json=variant.metadata or None,
        )
        self._var_repo.add(row)
        return row.id

    def get_variant(self, variant_id: str) -> KernelVariant | None:
        """Load a KernelVariant by variant_id, including its parent KernelSource."""
        row = self._var_repo.get_by_id(variant_id)
        if row is None:
            return None
        parent = self.get_source(row.parent_kernel_id)
        if parent is None:
            raise StorageError(
                f"Orphan variant '{variant_id}': parent kernel id={row.parent_kernel_id} not found"
            )
        return self._variant_to_pydantic(row, parent)

    def list_variants_for_kernel(
        self, kernel_id: int, offset: int = 0, limit: int = 100
    ) -> list[KernelVariant]:
        """All variants descending from a given kernel source."""
        stmt = (
            select(KernelVariantRow)
            .where(KernelVariantRow.parent_kernel_id == kernel_id)
            .order_by(KernelVariantRow.iteration)
            .offset(offset)
            .limit(limit)
        )
        rows = self._session.execute(stmt).scalars().all()
        parent = self.get_source(kernel_id)
        if parent is None:
            return []
        return [self._variant_to_pydantic(r, parent) for r in rows]

    # ── Public helpers ─────────────────────────────────────────

    def resolve_kernel_id(self, kernel: KernelSource) -> int:
        """Find the DB id for a kernel source. Inserts if not persisted yet.

        This is the public entry point for cross-repo kernel resolution.
        """
        stmt = select(KernelSourceRow.id).where(
            KernelSourceRow.name == kernel.name,
            KernelSourceRow.source_code == kernel.source_code,
        )
        existing = self._session.execute(stmt).scalar_one_or_none()
        if existing is not None:
            return existing
        return self.save_source(kernel)

    # ── Conversion helpers ─────────────────────────────────────

    def _source_to_pydantic(self, row: KernelSourceRow) -> KernelSource:
        return KernelSource(
            name=row.name,
            language=KernelLanguage(row.language),
            source_code=row.source_code,
            file_path=row.file_path,
            entry_point=row.entry_point,
            submitted_at=row.submitted_at,
            metadata=row.metadata_json or {},
        )

    def _variant_to_pydantic(
        self, row: KernelVariantRow, parent: KernelSource
    ) -> KernelVariant:
        history = from_json_list(row.transform_history_json, TransformSpec)
        return KernelVariant(
            variant_id=row.id,
            parent_source=parent,
            source_code=row.source_code,
            language=KernelLanguage(row.language),
            entry_point=row.entry_point,
            transform_history=history,
            iteration=row.iteration,
            created_at=row.created_at,
            metadata=row.metadata_json or {},
        )