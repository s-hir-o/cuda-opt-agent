"""Feature normalization — fill missing values with safe defaults."""

from __future__ import annotations

from cuda_opt_agent.schemas.features import KernelFeatureVector

# Defaults for None fields. Derived from "neutral" values that don't bias models.
_DEFAULTS: dict[str, float | bool | int] = {
    # Instruction mix: assume balanced if unknown
    "frac_alu": 0.3,
    "frac_fma": 0.2,
    "frac_memory_load": 0.2,
    "frac_memory_store": 0.1,
    "frac_shared": 0.0,
    "frac_control_flow": 0.05,
    # Parallelism
    "ilp_estimate": 1.0,
    "critical_path_length": 1,
    "total_instructions": 1,
    # Memory — assume optimistic
    "is_coalesced": True,
    "has_shared_memory": False,
    "bank_conflict_count": 0,
    # Occupancy — assume moderate
    "achieved_occupancy": 0.5,
    "registers_per_thread": 32,
    "occupancy_limited_by_registers": False,
    "occupancy_limited_by_shared_mem": False,
    # Compute
    "sm_throughput_pct": 50.0,
    "memory_throughput_pct": 50.0,
    "has_divergent_branches": False,
    # Launch config
    "block_size_x": 256,
    "block_size_y": 1,
    "block_size_z": 1,
    "grid_size_x": 1,
    "grid_size_y": 1,
    "grid_size_z": 1,
    "loop_nest_depth": 1,
}


def fill_defaults(vector: KernelFeatureVector) -> KernelFeatureVector:
    """Return a new vector with None fields filled from safe defaults."""
    data = vector.model_dump()
    for key, default in _DEFAULTS.items():
        if data.get(key) is None:
            data[key] = default
    return KernelFeatureVector(**data)
