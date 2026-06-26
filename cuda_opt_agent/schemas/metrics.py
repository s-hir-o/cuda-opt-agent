"""Profiling metrics from NVIDIA Nsight Compute."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class ProfileMetrics(BaseModel):
    """Parsed NCU profiling output for a single kernel invocation."""

    kernel_name: str
    # Add this field to ProfileMetrics, near the top with the other fields:
    variant_id: str | None = None
    gpu_name: str = ""
    arch: str = ""


    raw_metrics: dict[str, Any] = {}
    metadata: dict[str, Any] | None = None  # <-- ADD THIS LINE
    # Timing
    gpu_time_ns: float | None = None
    gpu_time_us: float | None = None

    # Throughput (% of peak)
    sm_throughput: float | None = None
    dram_throughput: float | None = None
    l1_throughput: float | None = None
    l2_throughput: float | None = None

    # Occupancy
    achieved_occupancy: float | None = None
    theoretical_occupancy: float | None = None

    # Launch config
    registers_per_thread: int | None = None
    block_size: tuple[int, ...] | None = None
    grid_size: tuple[int, ...] | None = None
    shared_mem_bytes: int | None = None

    # Memory
    global_load_efficiency: float | None = None
    global_store_efficiency: float | None = None
    l2_hit_rate: float | None = None
    l1_hit_rate: float | None = None

    # Occupancy limiters
    occupancy_limited_by_registers: bool = False
    occupancy_limited_by_shared_mem: bool = False
    occupancy_limited_by_block_size: bool = False

    # Raw metrics (unparsed NCU output)
    raw_metrics: dict[str, Any] = Field(default_factory=dict)

    profiled_at: datetime = Field(default_factory=datetime.utcnow)

    @property
    def is_bounded_by_memory(self) -> bool:
        return (self.dram_throughput or 0) > (self.sm_throughput or 0)

    @property
    def is_bounded_by_compute(self) -> bool:
        return (self.sm_throughput or 0) > (self.dram_throughput or 0)
