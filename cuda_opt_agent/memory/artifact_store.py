"""File-based artifact storage for kernel sources, PTX, cubin, and other files.

Organized by run_id with content-addressable deduplication.
Not stored in the database — filesystem only.
"""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

from cuda_opt_agent.core.exceptions import ArtifactNotFoundError


class ArtifactStore:
    """Manages file artifacts (source, PTX, cubin) on the local filesystem.

    Directory layout:
        {base_dir}/{run_id}/{kind}/{content_hash}.{ext}
    """

    def __init__(self, base_dir: str | Path) -> None:
        self._base_dir = Path(base_dir)
        self._base_dir.mkdir(parents=True, exist_ok=True)

    @property
    def base_dir(self) -> Path:
        return self._base_dir

    # ── Store ──────────────────────────────────────────────────

    def store(
        self,
        run_id: str,
        name: str,
        content: str | bytes,
        kind: str = "source",
    ) -> str:
        """Store an artifact and return its relative path.

        Uses content-hash for deduplication: identical content within the same
        run_id+kind produces the same path.

        Args:
            run_id: The optimization run this artifact belongs to.
            name: Original filename (used for extension detection).
            content: File content as string or bytes.
            kind: Subdirectory category (source, ptx, cubin, metrics, etc.).

        Returns:
            Relative path from base_dir to the stored artifact.
        """
        if isinstance(content, str):
            content_bytes = content.encode("utf-8")
        else:
            content_bytes = content

        content_hash = hashlib.sha256(content_bytes).hexdigest()[:16]
        ext = Path(name).suffix or ".bin"
        filename = f"{content_hash}{ext}"

        dest_dir = self._base_dir / run_id / kind
        dest_dir.mkdir(parents=True, exist_ok=True)

        dest_path = dest_dir / filename
        # Skip write if identical content already exists
        if not dest_path.exists() or dest_path.read_bytes() != content_bytes:
            dest_path.write_bytes(content_bytes)

        return str(dest_path.relative_to(self._base_dir))

    # ── Retrieve ───────────────────────────────────────────────

    def retrieve(self, rel_path: str) -> bytes:
        """Read artifact bytes by relative path. Raises ArtifactNotFoundError."""
        full_path = self._base_dir / rel_path
        if not full_path.is_file():
            raise ArtifactNotFoundError(f"Artifact not found: {rel_path}")
        return full_path.read_bytes()

    def retrieve_text(self, rel_path: str) -> str:
        """Read artifact as UTF-8 text."""
        return self.retrieve(rel_path).decode("utf-8")

    # ── List / Query ───────────────────────────────────────────

    def list_artifacts(self, run_id: str) -> list[str]:
        """List all artifact paths for a run, sorted."""
        run_dir = self._base_dir / run_id
        if not run_dir.is_dir():
            return []
        return sorted(
            str(p.relative_to(self._base_dir))
            for p in run_dir.rglob("*")
            if p.is_file()
        )

    def list_kinds(self, run_id: str) -> list[str]:
        """List artifact kinds (subdirectories) for a run."""
        run_dir = self._base_dir / run_id
        if not run_dir.is_dir():
            return []
        return sorted(
            p.name for p in run_dir.iterdir() if p.is_dir()
        )

    def exists(self, rel_path: str) -> bool:
        """Check whether an artifact exists."""
        return (self._base_dir / rel_path).is_file()


    def delete(self, rel_path: str) -> None:
        """Delete a single artifact by relative path.

        No-op if the file does not exist. Raises ArtifactNotFoundError
        is NOT raised for missing files - this is a cleanup operation.
        """
        full_path = self._base_dir / rel_path
        if full_path.is_file():
            full_path.unlink()

    def get_size(self, run_id: str) -> int:
        """Total bytes of all artifacts for a run."""
        run_dir = self._base_dir / run_id
        if not run_dir.is_dir():
            return 0
        return sum(p.stat().st_size for p in run_dir.rglob("*") if p.is_file())

    # ── Cleanup ────────────────────────────────────────────────

    def cleanup(self, run_id: str) -> None:
        """Delete all artifacts for a run."""
        run_dir = self._base_dir / run_id
        if run_dir.is_dir():
            shutil.rmtree(run_dir)

    def cleanup_all(self) -> None:
        """Delete all artifacts. Use with caution."""
        if self._base_dir.is_dir():
            shutil.rmtree(self._base_dir)
            self._base_dir.mkdir(parents=True, exist_ok=True)
