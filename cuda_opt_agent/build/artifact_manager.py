"""Compilation artifact manager.

Tracks artifacts produced by NVCCRunner across an optimization run.
Provides lookup by variant_id, cleanup, and disk usage reporting.
Wraps the lower-level ArtifactStore from memory/.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from cuda_opt_agent.memory.artifact_store import ArtifactStore
from cuda_opt_agent.schemas.compilation import CompilationResult


@dataclass
class ArtifactRecord:
    """Metadata for a tracked compilation artifact."""

    variant_id: str
    kind: str  # "source", "cubin", "ptx", "executable"
    rel_path: str
    arch: str = ""
    compile_time_s: float = 0.0
    success: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


class ArtifactManager:
    """High-level artifact management for an optimization run.

    Wraps ArtifactStore and adds:
    - Variant-aware tracking (not just run-level)
    - CompilationResult integration (auto-stores PTX, cubin, executable)
    - Standalone store methods for individual artifact types
    - Lookup by variant_id
    - Disk usage reporting
    """

    def __init__(self, artifact_store: ArtifactStore, run_id: str) -> None:
        self._store = artifact_store
        self._run_id = run_id
        self._records: dict[str, list[ArtifactRecord]] = {}

    @property
    def run_id(self) -> str:
        return self._run_id

    @property
    def store(self) -> ArtifactStore:
        return self._store

    # ── Store individual artifacts ─────────────────────────────

    def store_source(
        self, variant_id: str, source: str, filename: str = "kernel.cu"
    ) -> str:
        """Store kernel source code. Returns relative path."""
        rel = self._store.store(
            self._run_id, filename, content=source, kind="source"
        )
        self._add_record(variant_id, ArtifactRecord(
            variant_id=variant_id, kind="source", rel_path=rel,
        ))
        return rel

    def store_ptx(self, variant_id: str, ptx_source: str) -> str:
        """Store PTX source text. Returns relative path."""
        rel = self._store.store(
            self._run_id, "kernel.ptx", content=ptx_source, kind="ptx"
        )
        self._add_record(variant_id, ArtifactRecord(
            variant_id=variant_id, kind="ptx", rel_path=rel,
        ))
        return rel

    def store_cubin(
        self, variant_id: str, cubin_path: str, arch: str = ""
    ) -> str | None:
        """Store a compiled cubin file. Returns relative path or None if file missing."""
        p = Path(cubin_path)
        if not p.is_file():
            return None
        rel = self._store.store(
            self._run_id, p.name, content=p.read_bytes(), kind="cubin"
        )
        self._add_record(variant_id, ArtifactRecord(
            variant_id=variant_id, kind="cubin", rel_path=rel, arch=arch,
        ))
        return rel

    def store_executable(
        self, variant_id: str, exe_path: str, arch: str = ""
    ) -> str | None:
        """Store a compiled executable for profiling. Returns relative path or None."""
        p = Path(exe_path)
        if not p.is_file():
            return None
        rel = self._store.store(
            self._run_id, p.name, content=p.read_bytes(), kind="executable"
        )
        self._add_record(variant_id, ArtifactRecord(
            variant_id=variant_id, kind="executable", rel_path=rel, arch=arch,
        ))
        return rel

    # ── Store from CompilationResult ───────────────────────────

    def store_compilation_result(
        self, variant_id: str, result: CompilationResult
    ) -> list[str]:
        """Store all artifacts from a CompilationResult.

        Dispatches based on which path fields are populated:
        - cubin_path -> stores as kind="cubin"
        - binary_path -> stores as kind="executable"
        - ptx_source -> stores as kind="ptx"
        - source_code -> always stores as kind="source"

        Returns list of relative paths for all stored artifacts.
        """
        paths: list[str] = []

        # Source (always)
        paths.append(self.store_source(variant_id, result.source_code))

        # Cubin (kernel binary)
        if result.cubin_path:
            cubin_rel = self.store_cubin(
                variant_id, result.cubin_path, arch=result.arch
            )
            if cubin_rel:
                for rec in self._records.get(variant_id, []):
                    if rec.rel_path == cubin_rel:
                        rec.compile_time_s = result.compile_time_s
                        rec.success = result.success
                paths.append(cubin_rel)

        # Executable (profiling binary — distinct from cubin)
        if result.binary_path:
            exe_rel = self.store_executable(
                variant_id, result.binary_path, arch=result.arch
            )
            if exe_rel:
                for rec in self._records.get(variant_id, []):
                    if rec.rel_path == exe_rel:
                        rec.compile_time_s = result.compile_time_s
                        rec.success = result.success
                paths.append(exe_rel)

        # PTX
        if result.ptx_source:
            paths.append(self.store_ptx(variant_id, result.ptx_source))

        return paths

    # ── Retrieve ───────────────────────────────────────────────

    def get_ptx(self, variant_id: str) -> str | None:
        """Retrieve the PTX source for a variant, or None."""
        rec = self._find_record(variant_id, "ptx")
        if rec is None:
            return None
        try:
            return self._store.retrieve_text(rec.rel_path)
        except Exception:
            return None

    def get_source(self, variant_id: str) -> str | None:
        """Retrieve the source code for a variant, or None."""
        rec = self._find_record(variant_id, "source")
        if rec is None:
            return None
        try:
            return self._store.retrieve_text(rec.rel_path)
        except Exception:
            return None

    def get_executable_path(self, variant_id: str) -> str | None:
        """Retrieve the full path to a compiled executable, or None."""
        rec = self._find_record(variant_id, "executable")
        if rec is None:
            return None
        full = self._store.base_dir / rec.rel_path
        return str(full) if full.is_file() else None

    def get_records(self, variant_id: str) -> list[ArtifactRecord]:
        """All artifact records for a variant."""
        return list(self._records.get(variant_id, []))

    # ── Cleanup ────────────────────────────────────────────────

    def cleanup_variant(self, variant_id: str) -> None:
        """Remove all tracked artifacts for a variant from disk.

        Uses ArtifactStore.delete() for consistent path resolution.
        """
        records = self._records.get(variant_id, [])
        referenced_elsewhere = {
            rec.rel_path
            for other_variant, other_records in self._records.items()
            if other_variant != variant_id
            for rec in other_records
        }
        for rec in records:
            if rec.rel_path not in referenced_elsewhere:
                self._store.delete(rec.rel_path)
        self._records.pop(variant_id, None)

    def cleanup_run(self) -> None:
        """Remove all artifacts for this run from disk."""
        self._store.cleanup(self._run_id)
        self._records.clear()

    # ── Reporting ──────────────────────────────────────────────

    def total_size_bytes(self) -> int:
        """Total disk usage for this run's artifacts."""
        return self._store.get_size(self._run_id)

    def variant_count(self) -> int:
        """Number of tracked variants."""
        return len(self._records)

    def artifact_count(self) -> int:
        """Total number of tracked artifact files."""
        return sum(len(recs) for recs in self._records.values())

    def summary(self) -> dict[str, Any]:
        """Summary of all tracked artifacts."""
        return {
            "run_id": self._run_id,
            "variant_count": self.variant_count(),
            "artifact_count": self.artifact_count(),
            "total_size_bytes": self.total_size_bytes(),
            "variants": {
                vid: [
                    {"kind": r.kind, "path": r.rel_path, "success": r.success}
                    for r in recs
                ]
                for vid, recs in self._records.items()
            },
        }

    # ── Internal ───────────────────────────────────────────────

    def _add_record(self, variant_id: str, record: ArtifactRecord) -> None:
        self._records.setdefault(variant_id, []).append(record)

    def _find_record(self, variant_id: str, kind: str) -> ArtifactRecord | None:
        for rec in self._records.get(variant_id, []):
            if rec.kind == kind:
                return rec
        return None
