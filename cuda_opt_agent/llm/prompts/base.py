"""Shared formatting utilities for building LLM prompts from Pydantic models.

Each function takes a typed Pydantic model and returns a clean,
LLM-readable text block. Keeps prompt construction DRY.
"""

from __future__ import annotations

from cuda_opt_agent.schemas.bottleneck import BottleneckResult
from cuda_opt_agent.schemas.kernel import KernelSource, KernelVariant
from cuda_opt_agent.schemas.metrics import ProfileMetrics


def format_kernel_source(kernel: KernelSource | KernelVariant) -> str:
    """Format kernel source code for inclusion in a prompt."""
    name = kernel.name if isinstance(kernel, KernelSource) else kernel.variant_id
    lang = kernel.language.value
    return (
        f"## Kernel: {name} ({lang})\n"
        f"Entry point: `{kernel.entry_point}`\n\n"
        f"```{lang}\n{kernel.source_code}\n```"
    )


def format_bottleneck(bottleneck: BottleneckResult) -> str:
    """Format bottleneck classification for a prompt."""
    lines = [
        f"**Primary bottleneck**: {bottleneck.primary.value} "
        f"(confidence: {bottleneck.confidence:.0%})",
    ]
    if bottleneck.secondary:
        lines.append(f"**Secondary**: {bottleneck.secondary.value}")
    if bottleneck.reasoning:
        lines.append(f"**Reasoning**: {bottleneck.reasoning}")
    if bottleneck.recommended_transforms:
        lines.append(
            f"**Suggested transforms**: {', '.join(bottleneck.recommended_transforms)}"
        )
    return "\n".join(lines)


def format_metrics(metrics: ProfileMetrics) -> str:
    """Format key profiling metrics for a prompt."""
    lines = ["## Profiling Metrics\n"]
    if metrics.gpu_time_us is not None:
        lines.append(f"- GPU time: {metrics.gpu_time_us:.2f} µs")
    if metrics.sm_throughput is not None:
        lines.append(f"- SM throughput: {metrics.sm_throughput:.1f}% of peak")
    if metrics.dram_throughput is not None:
        lines.append(f"- DRAM throughput: {metrics.dram_throughput:.1f}% of peak")
    if metrics.achieved_occupancy is not None:
        lines.append(f"- Achieved occupancy: {metrics.achieved_occupancy:.1%}")
    if metrics.registers_per_thread is not None:
        lines.append(f"- Registers/thread: {metrics.registers_per_thread}")
    if metrics.shared_mem_bytes is not None:
        lines.append(f"- Shared memory: {metrics.shared_mem_bytes} bytes")
    if metrics.block_size is not None:
        lines.append(f"- Block size: {tuple(metrics.block_size)}")
    if metrics.grid_size is not None:
        lines.append(f"- Grid size: {tuple(metrics.grid_size)}")
    limiters = []
    if metrics.occupancy_limited_by_registers:
        limiters.append("registers")
    if metrics.occupancy_limited_by_shared_mem:
        limiters.append("shared memory")
    if metrics.occupancy_limited_by_block_size:
        limiters.append("block size")
    if limiters:
        lines.append(f"- Occupancy limited by: {', '.join(limiters)}")

    eff_parts = []
    if metrics.global_load_efficiency is not None:
        eff_parts.append(f"load={metrics.global_load_efficiency:.1f}%")
    if metrics.global_store_efficiency is not None:
        eff_parts.append(f"store={metrics.global_store_efficiency:.1f}%")
    if eff_parts:
        lines.append(f"- Global memory efficiency: {', '.join(eff_parts)}")

    return "\n".join(lines)


def format_variant_diff(variant: KernelVariant) -> str:
    """Format transform history of a variant."""
    if not variant.transform_history:
        return "**No transforms applied yet** (original kernel)."

    lines = ["## Transform History\n"]
    for i, spec in enumerate(variant.transform_history, 1):
        params = ", ".join(f"{k}={v}" for k, v in spec.params.items())
        line = f"{i}. `{spec.transform_name}` ({spec.backend.value} backend)"
        if params:
            line += f" — params: {params}"
        if spec.reason:
            line += f"\n   Reason: {spec.reason}"
        lines.append(line)
    return "\n".join(lines)
