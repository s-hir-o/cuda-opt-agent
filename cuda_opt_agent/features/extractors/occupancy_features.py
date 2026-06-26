"""Occupancy analysis features — derived from metrics and analysis."""

from __future__ import annotations

from cuda_opt_agent.features.base import FeatureExtractor
from cuda_opt_agent.schemas.analysis import KernelAnalysis
from cuda_opt_agent.schemas.features import KernelFeatureVector
from cuda_opt_agent.schemas.kernel import KernelSource, KernelVariant
from cuda_opt_agent.schemas.metrics import ProfileMetrics

# Peak FP32 TFLOPS and bandwidth per architecture
_GPU_SPECS: dict[str, dict] = {
    "sm_80": {"peak_fp32_tflops": 19.5, "peak_bw_gbs": 2039, "name": "A100"},
    "sm_89": {"peak_fp32_tflops": 82.6, "peak_bw_gbs": 1008, "name": "RTX 4090"},
    "sm_90": {"peak_fp32_tflops": 67.0, "peak_bw_gbs": 3350, "name": "H100"},
}


class OccupancyFeatureExtractor(FeatureExtractor):

    @property
    def name(self) -> str:
        return "occupancy"

    @property
    def required_inputs(self) -> list[str]:
        return ["metrics"]  # optional: "analysis"

    def extract(
        self,
        kernel: KernelSource | KernelVariant,
        metrics: ProfileMetrics | None,
        analysis: KernelAnalysis | None,
    ) -> KernelFeatureVector:
        if metrics is None:
            return KernelFeatureVector()

        return KernelFeatureVector(
            occupancy_limited_by_registers=metrics.occupancy_limited_by_registers,
            occupancy_limited_by_shared_mem=metrics.occupancy_limited_by_shared_mem,
            registers_per_thread=metrics.registers_per_thread,
            achieved_occupancy=metrics.achieved_occupancy,
            has_shared_memory=(
                analysis.has_shared_memory if analysis else
                (metrics.shared_mem_bytes or 0) > 0
            ),
        )
