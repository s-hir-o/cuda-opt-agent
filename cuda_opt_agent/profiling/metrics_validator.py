



"""Validate and normalize ProfileMetrics after parsing."""

from __future__ import annotations

from cuda_opt_agent.schemas.metrics import ProfileMetrics


def validate_metrics(metrics: ProfileMetrics) -> list[str]:
    """Check metrics for suspicious values. Returns list of warnings."""
    warnings: list[str] = []

    if metrics.gpu_time_us is not None and metrics.gpu_time_us <= 0:
        warnings.append(f"Non-positive GPU time: {metrics.gpu_time_us}")

    for name, val in [
        ("sm_throughput", metrics.sm_throughput),
        ("dram_throughput", metrics.dram_throughput),
        ("l1_throughput", metrics.l1_throughput),
        ("l2_throughput", metrics.l2_throughput),
    ]:
        if val is not None and not (0 <= val <= 100):
            warnings.append(f"{name} out of range [0,100]: {val}")

    if metrics.achieved_occupancy is not None:
        if not (0 <= metrics.achieved_occupancy <= 1.0):
            warnings.append(
                f"Achieved occupancy out of range [0,1]: {metrics.achieved_occupancy}"
            )

    if metrics.registers_per_thread is not None:
        if not (1 <= metrics.registers_per_thread <= 255):
            warnings.append(f"Suspicious register count: {metrics.registers_per_thread}")

    for name, val in [
        ("global_load_efficiency", metrics.global_load_efficiency),
        ("global_store_efficiency", metrics.global_store_efficiency),
        ("l1_hit_rate", metrics.l1_hit_rate),
        ("l2_hit_rate", metrics.l2_hit_rate),
    ]:
        if val is not None and not (0 <= val <= 100):
            warnings.append(f"{name} out of range [0,100]: {val}")

    return warnings

