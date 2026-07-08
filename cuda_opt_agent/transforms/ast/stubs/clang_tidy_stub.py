"""Stub for future clang-tidy custom CUDA checks. Import-safe — raises on use."""

from __future__ import annotations
from typing import Any

from cuda_opt_agent.schemas.bottleneck import BottleneckClass, BottleneckResult
from cuda_opt_agent.schemas.kernel import KernelSource, KernelVariant
from cuda_opt_agent.schemas.transform import TransformResult
from cuda_opt_agent.transforms.ast.base import ASTTransformation


class ClangTidyStub(ASTTransformation):
    @property
    def name(self) -> str:
        return "clang_tidy_cuda"

    @property
    def targets(self) -> list[BottleneckClass]:
        return list(BottleneckClass)

    @property
    def parameter_schema(self) -> dict:
        return {"type": "object", "properties": {}}

    def is_applicable(self, kernel: KernelSource | KernelVariant, bottleneck: BottleneckResult, params: dict[str, Any]) -> bool:
        return False

    def parse_ast(self, source: str) -> Any:
        raise NotImplementedError("clang-tidy integration not implemented.")

    def apply_to_ast(self, ast: Any, params: dict[str, Any]) -> Any:
        raise NotImplementedError

    def emit_source(self, ast: Any) -> str:
        raise NotImplementedError

    def validate(self, result: TransformResult) -> bool:
        return False