"""Transform specification, result, and backend enum."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class TransformBackend(str, Enum):
    TEXT = "text"    # Regex/string-based source transforms (current)
    AST = "ast"      # Clang AST-based transforms (future)
    PTX = "ptx"      # PTX instruction-level transforms (future)
    LLVM = "llvm"    # LLVM IR transforms (future)


class TransformSpec(BaseModel):
    """Specification of a transform to apply, including parameters."""

    transform_name: str
    backend: TransformBackend = TransformBackend.TEXT
    params: dict[str, Any] = Field(default_factory=dict)
    reason: str = ""
    source: str = "agent"  # "agent" | "llm" | "search" | "manual"

    def __str__(self) -> str:
        param_str = ", ".join(f"{k}={v}" for k, v in self.params.items())
        return f"{self.transform_name}({param_str})"


class TransformResult(BaseModel):
    """Result of applying a transform to a kernel."""

    spec: TransformSpec
    original_source: str
    transformed_source: str
    success: bool = True
    error_message: str | None = None
    warnings: list[str] = Field(default_factory=list)
    diff_summary: str = ""
    applied_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def changed(self) -> bool:
        return self.transformed_source != self.original_source
