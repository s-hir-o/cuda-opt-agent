"""Compilation result and diagnostics."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class DiagnosticSeverity(str, Enum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"
    REMARK = "remark"


class CompilationDiagnostic(BaseModel):
    """A single diagnostic from nvcc output."""

    severity: DiagnosticSeverity
    message: str
    file: str | None = None
    line: int | None = None
    column: int | None = None
    raw: str = ""

    @property
    def is_error(self) -> bool:
        return self.severity == DiagnosticSeverity.ERROR

    @property
    def is_warning(self) -> bool:
        return self.severity == DiagnosticSeverity.WARNING


class CompilationResult(BaseModel):
    """Result of compiling a kernel with nvcc."""

    success: bool
    source_code: str
    arch: str
    cubin_path: str | None = None
    ptx_path: str | None = None
    ptx_source: str | None = None
    diagnostics: list[CompilationDiagnostic] = Field(default_factory=list)
    compile_time_s: float = 0.0
    compiled_at: datetime = Field(default_factory=datetime.utcnow)
    stdout: str = ""
    stderr: str = ""
    returncode: int = 0

    @property
    def errors(self) -> list[CompilationDiagnostic]:
        return [d for d in self.diagnostics if d.is_error]

    @property
    def warnings(self) -> list[CompilationDiagnostic]:
        return [d for d in self.diagnostics if d.is_warning]

    @property
    def has_register_spill(self) -> bool:
        return any("spill" in d.message.lower() for d in self.warnings)
