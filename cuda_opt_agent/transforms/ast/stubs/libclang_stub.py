"""Stub for future libclang AST transforms. Import-safe — raises on use."""

from __future__ import annotations
from typing import Any

from cuda_opt_agent.schemas.bottleneck import BottleneckClass, BottleneckResult
from cuda_opt_agent.schemas.kernel import KernelSource, KernelVariant
from cuda_opt_agent.schemas.transform import TransformResult
from cuda_opt_agent.transforms.ast.base import ASTTransformation


class LibClangLoopUnrollStub(ASTTransformation):
    @property
    def name(self) -> str:
        return "libclang_loop_unroll"

    @property
    def targets(self) -> list[BottleneckClass]:
        return [BottleneckClass.COMPUTE_BOUND, BottleneckClass.LATENCY_BOUND]

    @property
    def parameter_schema(self) -> dict:
        return {"type": "object", "properties": {"factor": {"type": "integer", "default": 4}}}

    def is_applicable(self, kernel: KernelSource | KernelVariant, bottleneck: BottleneckResult, params: dict[str, Any]) -> bool:
        return False

    def parse_ast(self, source: str) -> Any:
        raise NotImplementedError("libclang AST transforms not implemented.")

    def apply_to_ast(self, ast: Any, params: dict[str, Any]) -> Any:
        raise NotImplementedError

    def emit_source(self, ast: Any) -> str:
        raise NotImplementedError

    def validate(self, result: TransformResult) -> bool:
        return False


TEST FILES: