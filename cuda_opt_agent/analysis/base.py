"""Abstract kernel analyzer — backend-agnostic analysis interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

from cuda_opt_agent.schemas.analysis import KernelAnalysis
from cuda_opt_agent.schemas.kernel import KernelSource, KernelVariant


class KernelAnalyzer(ABC):
    """Analyze a kernel and return structured KernelAnalysis.

    Current backend: PTX text parsing.
    Future backends: LLVM IR, Clang AST, SASS disassembly.
    """

    @abstractmethod
    async def analyze(
        self,
        kernel: KernelSource | KernelVariant,
        ptx_source: str | None,
        run_id: str,
    ) -> KernelAnalysis:
        """Run full analysis. ptx_source=None skips PTX-level analysis."""

    @abstractmethod
    async def is_available(self) -> bool:
        """Check if required tools/libraries are present."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Analyzer identifier (e.g., 'ptx', 'llvm')."""
