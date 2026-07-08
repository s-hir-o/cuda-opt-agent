"""AST Transformation base class — extends Transformation for AST-backed transforms.

Orchestration code (coordinator) never changes — it calls apply() on the same
Transformation ABC. Only the backend property and internal implementation differ.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any

from cuda_opt_agent.schemas.kernel import KernelSource, KernelVariant
from cuda_opt_agent.schemas.transform import TransformBackend, TransformResult
from cuda_opt_agent.transforms.base import Transformation


class ASTTransformation(Transformation):
    """Base for AST-based transforms (Clang libclang, tree-sitter, etc.)."""

    @property
    def backend(self) -> TransformBackend:
        return TransformBackend.AST

    @abstractmethod
    def parse_ast(self, source: str) -> Any:
        """Parse source to an AST representation. Backend-specific."""

    @abstractmethod
    def apply_to_ast(self, ast: Any, params: dict[str, Any]) -> Any:
        """Perform the transform on the AST. Backend-specific."""

    @abstractmethod
    def emit_source(self, ast: Any) -> str:
        """Emit source code from the transformed AST."""

    def apply(
        self,
        kernel: KernelSource | KernelVariant,
        params: dict[str, Any],
    ) -> TransformResult:
        """Default implementation: parse → transform → emit → wrap."""
        try:
            ast = self.parse_ast(kernel.source_code)
            transformed_ast = self.apply_to_ast(ast, params)
            output = self.emit_source(transformed_ast)
            return self._make_result(kernel, output, params, f"{self.name} (AST backend)")
        except NotImplementedError:
            raise
        except Exception as exc:
            return TransformResult(
                spec=self._make_spec(params),
                original_source=kernel.source_code,
                transformed_source=kernel.source_code,
                success=False,
                error_message=f"{self.name} AST transform failed: {exc}",
            )

    def _make_spec(self, params: dict[str, Any]):
        from cuda_opt_agent.schemas.transform import TransformSpec
        return TransformSpec(
            transform_name=self.name,
            backend=TransformBackend.AST,
            params=params,
        )