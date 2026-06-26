"""Stub for future Clang AST analysis. Import-safe — raises on use."""

from __future__ import annotations

from cuda_opt_agent.analysis.base import KernelAnalyzer
from cuda_opt_agent.schemas.analysis import KernelAnalysis
from cuda_opt_agent.schemas.kernel import KernelSource, KernelVariant


class ClangASTAnalyzer(KernelAnalyzer):

    @property
    def name(self) -> str:
        return "clang_ast"

    async def is_available(self) -> bool:
        return False

    async def analyze(
        self, kernel: KernelSource | KernelVariant, ptx_source: str | None, run_id: str
    ) -> KernelAnalysis:
        raise NotImplementedError(
            "Clang AST analysis not yet implemented. See docs/LLVM_CLANG_ROADMAP.md."
        )
