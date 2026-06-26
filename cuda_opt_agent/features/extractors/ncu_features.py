"""Extract features from ProfileMetrics (NCU output)."""

from __future__ import annotations

from cuda_opt_agent.features.base import FeatureExtractor
from cuda_opt_agent.schemas.analysis import KernelAnalysis
from cuda_opt_agent.schemas.features import KernelFeatureVector
from cuda_opt_agent.schemas.kernel import KernelSource, KernelVariant
from cuda_opt_agent.schemas.metrics import ProfileMetrics


class NCUFeatureExtractor(FeatureExtractor):

    @property
    def name(self) -> str:
        return "ncu"

    @property
    def required_inputs(self) -> list[str]:
        return ["metrics"]

    def extract(
        self,
        kernel: KernelSource | KernelVariant,
        metrics: ProfileMetrics | None,
        analysis: KernelAnalysis | None,
    ) -> KernelFeatureVector:
        if metrics is None:
            return KernelFeatureVector()

        bs = metrics.block_size
        gs = metrics.grid_size

        return KernelFeatureVector(
            # Throughput
            sm_throughput_pct=metrics.sm_throughput,
            memory_throughput_pct=metrics.dram_throughput,
            # Occupancy
            achieved_occupancy=metrics.achieved_occupancy,
            registers_per_thread=metrics.registers_per_thread,
            occupancy_limited_by_registers=metrics.occupancy_limited_by_registers,
            occupancy_limited_by_shared_mem=metrics.occupancy_limited_by_shared_mem,
            # Memory
            is_coalesced=(
                (metrics.global_load_efficiency or 0) > 80.0
                if metrics.global_load_efficiency is not None
                else None
            ),
            # Launch config
            block_size_x=bs[0] if bs else None,
            block_size_y=bs[1] if bs and len(bs) > 1 else None,
            block_size_z=bs[2] if bs and len(bs) > 2 else None,
            grid_size_x=gs[0] if gs else None,
            grid_size_y=gs[1] if gs and len(gs) > 1 else None,
            grid_size_z=gs[2] if gs and len(gs) > 2 else None,
        )
