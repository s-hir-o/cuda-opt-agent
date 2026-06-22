"""Kernel feature vector for cost model input."""

from __future__ import annotations

from typing import Any

import numpy as np
from pydantic import BaseModel, Field


class KernelFeatureVector(BaseModel):
    """Numeric feature vector extracted from kernel analysis + profiling.

    Each feature is optional — absent data results in None, not 0.
    Cost models handle missing features gracefully.
    """

    # ── Instruction mix (fractions, sum to ~1.0) ──
    frac_alu: float | None = None
    frac_fma: float | None = None
    frac_memory_load: float | None = None
    frac_memory_store: float | None = None
    frac_shared: float | None = None
    frac_control_flow: float | None = None

    # ── Parallelism ──
    ilp_estimate: float | None = None
    critical_path_length: int | None = None
    total_instructions: int | None = None

    # ── Memory ──
    is_coalesced: bool | None = None
    has_shared_memory: bool | None = None
    bank_conflict_count: int | None = None
    stride_bytes: int | None = None
    memory_throughput_pct: float | None = None

    # ── Occupancy ──
    registers_per_thread: int | None = None
    achieved_occupancy: float | None = None
    occupancy_limited_by_registers: bool | None = None
    occupancy_limited_by_shared_mem: bool | None = None

    # ── Compute ──
    sm_throughput_pct: float | None = None
    has_divergent_branches: bool | None = None

    # ── Launch config ──
    block_size_x: int | None = None
    block_size_y: int | None = None
    block_size_z: int | None = None
    grid_size_x: int | None = None
    grid_size_y: int | None = None
    grid_size_z: int | None = None
    loop_nest_depth: int | None = None

    def to_numpy(self, fill: float = 0.0) -> np.ndarray:
        """Convert to numpy array, filling None with `fill`."""
        values = []
        for field_name in self.model_fields:
            val = getattr(self, field_name)
            if val is None:
                values.append(fill)
            elif isinstance(val, bool):
                values.append(1.0 if val else 0.0)
            else:
                values.append(float(val))
        return np.array(values, dtype=np.float32)

    def field_names(self) -> list[str]:
        return list(self.model_fields.keys())

    def available_count(self) -> int:
        return sum(1 for f in self.model_fields if getattr(self, f) is not None)

    model_config = {"arbitrary_types_allowed": True}
