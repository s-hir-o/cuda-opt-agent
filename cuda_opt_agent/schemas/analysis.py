"""PTX analysis models — v2 additions for Layer 3.5."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class OpcodeClass(str, Enum):
    """High-level instruction classification."""
    ALU = "alu"
    FMA = "fma"
    MEMORY_LOAD = "memory_load"
    MEMORY_STORE = "memory_store"
    SHARED_LOAD = "shared_load"
    SHARED_STORE = "shared_store"
    CONTROL_FLOW = "control_flow"
    CONVERSION = "conversion"
    COMPARISON = "comparison"
    ATOMIC = "atomic"
    SYNC = "sync"
    OTHER = "other"


class PTXInstruction(BaseModel):
    """A single parsed PTX instruction."""

    index: int
    raw: str
    opcode: str = ""
    opcode_class: OpcodeClass = OpcodeClass.OTHER
    operands: list[str] = Field(default_factory=list)
    predicate: str | None = None
    dest_registers: list[str] = Field(default_factory=list)
    src_registers: list[str] = Field(default_factory=list)
    memory_space: str | None = None  # "global", "shared", "local", etc.
    memory_offset: int | None = None
    is_branch: bool = False
    branch_target: int | None = None
    line_number: int | None = None


class DepType(str, Enum):
    RAW = "raw"  # Read After Write
    WAR = "war"  # Write After Read
    WAW = "waw"  # Write After Write


class DependencyEdge(BaseModel):
    """An edge in the instruction dependency graph."""

    from_idx: int
    to_idx: int
    dep_type: DepType
    register: str
    latency: int = 1  # estimated cycles


class DependencyGraph(BaseModel):
    """Full dependency graph for a PTX instruction sequence."""

    edges: list[DependencyEdge] = Field(default_factory=list)
    num_instructions: int = 0
    critical_path_length: int = 0
    estimated_ilp: float = 0.0

    @property
    def edge_count(self) -> int:
        return len(self.edges)

    def edges_from(self, idx: int) -> list[DependencyEdge]:
        return [e for e in self.edges if e.from_idx == idx]

    def edges_to(self, idx: int) -> list[DependencyEdge]:
        return [e for e in self.edges if e.to_idx == idx]


class MemAccessPattern(BaseModel):
    """Describes a detected memory access pattern."""

    pattern_type: str  # "stride", "sequential", "random", "broadcast"
    stride_bytes: int | None = None
    vector_width: int = 1
    is_coalesced: bool = True
    bank_conflicts: int = 0
    affected_instructions: list[int] = Field(default_factory=list)
    description: str = ""


class KernelAnalysis(BaseModel):
    """Complete analysis result for a kernel (PTX-level and source-level)."""

    instructions: list[PTXInstruction] = Field(default_factory=list)
    dependency_graph: DependencyGraph | None = None
    access_patterns: list[MemAccessPattern] = Field(default_factory=list)

    # Summary metrics
    total_instructions: int = 0
    instruction_mix: dict[str, int] = Field(default_factory=dict)
    register_count: int = 0
    critical_path_length: int = 0
    ilp_estimate: float = 0.0

    # Source-level observations
    loop_nest_depth: int = 0
    has_shared_memory: bool = False
    has_divergent_branches: bool = False
    has_uncoalesced_access: bool = False
    bank_conflict_count: int = 0

    # Metadata
    analysis_backend: str = "ptx"  # "ptx" | "llvm" | "clang" | "source"
    warnings: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
