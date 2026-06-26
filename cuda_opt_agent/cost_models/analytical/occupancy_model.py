
"""Occupancy-based cost model.

Predicts theoretical occupancy from launch parameters and estimates
performance impact of occupancy changes.
"""

from __future__ import annotations

from cuda_opt_agent.cost_models.base import CostModel
from cuda_opt_agent.schemas.cost_model import BoundType, CostEstimate, ModelPrediction
from cuda_opt_agent.schemas.features import KernelFeatureVector
from cuda_opt_agent.schemas.metrics import ProfileMetrics

# Per-architecture limits
_ARCH_OCCUPANCY: dict[str, dict[str, int]] = {
    "sm_80": {"max_threads": 2048, "max_regs": 65536, "max_smem": 164 * 1024, "warp_size": 32},
    "sm_89": {"max_threads": 1536, "max_regs": 65536, "max_smem": 100 * 1024, "warp_size": 32},
    "sm_90": {"max_threads": 2048, "max_regs": 65536, "max_smem": 228 * 1024, "warp_size": 32},
}
_DEFAULT_ARCH = "sm_80"


class OccupancyModel(CostModel):

    def __init__(self, arch: str = _DEFAULT_ARCH) -> None:
        self._arch = arch
        self._spec = _ARCH_OCCUPANCY.get(arch, _ARCH_OCCUPANCY[_DEFAULT_ARCH])

    @property
    def name(self) -> str:
        return "occupancy"

    def predict(
        self,
        features: KernelFeatureVector,
        metrics: ProfileMetrics | None,
    ) -> CostEstimate:
        try:
            return self._predict_inner(features, metrics)
        except Exception:
            return CostEstimate(
                model_name=self.name,
                confidence=0.0,
                reasoning="Insufficient data for occupancy prediction",
            )

    def _predict_inner(
        self,
        features: KernelFeatureVector,
        metrics: ProfileMetrics | None,
    ) -> CostEstimate:
        regs = features.registers_per_thread or 32
        threads = _block_threads(features)
        shared_mem = 0
        if metrics and metrics.shared_mem_bytes:
            shared_mem = metrics.shared_mem_bytes

        # Theoretical occupancy
        limiters = self._compute_limiters(regs, threads, shared_mem)

        theoretical_occ = 0.0
        limiting_factor = "unknown"
        if limiters:
            theoretical_occ = min(limiters.values())
            limiting_factor = min(limiters, key=limiters.get)

        achieved = features.achieved_occupancy
        confidence = 0.6
        if achieved is not None:
            # Compare theoretical vs achieved
            if abs(theoretical_occ - achieved) < 0.05:
                confidence = 0.8  # Model matches reality
            else:
                confidence = 0.4  # Gap suggests unknown factors

        return CostEstimate(
            model_name=self.name,
            bound_type=BoundType.OCCUPANCY if theoretical_occ < 0.75 else BoundType.UNKNOWN,
            predicted_occupancy=theoretical_occ,
            predicted_register_usage=regs,
            confidence=confidence,
            reasoning=f"Theoretical occupancy {theoretical_occ:.0%} limited by {limiting_factor}",
        )

    def predict_speedup(
        self,
        baseline: KernelFeatureVector,
        variant: KernelFeatureVector,
    ) -> ModelPrediction:
        b_occ = baseline.achieved_occupancy or self._theoretical_occ(baseline)
        v_occ = variant.achieved_occupancy or self._theoretical_occ(variant)

        if b_occ > 0:
            speedup = min(v_occ / b_occ, 2.0)
        else:
            speedup = 1.0

        improvements = []
        if v_occ > b_occ:
            improvements.append(f"Occupancy {b_occ:.0%} → {v_occ:.0%}")

        return ModelPrediction(
            model_name=self.name,
            predicted_speedup=round(speedup, 3),
            confidence=0.4,
            reasoning="Occupancy-based prediction (approximate)",
            improvement_areas=improvements,
        )

    def _theoretical_occ(self, features: KernelFeatureVector) -> float:
        regs = features.registers_per_thread or 32
        threads = _block_threads(features)
        limiters = self._compute_limiters(regs, threads, 0)
        return min(limiters.values()) if limiters else 1.0

    def _compute_limiters(self, regs_per_thread: int, block_threads: int, shared_mem: int) -> dict[str, float]:
        s = self._spec
        if block_threads == 0:
            block_threads = 256

        limiters: dict[str, float] = {}

        # Each limiter computes the occupancy fraction achievable by that resource alone.
        # occupancy = (max_blocks_per_sm * block_threads) / max_threads_per_sm

        # Thread limiter
        max_blocks_by_threads = s["max_threads"] // block_threads
        limiters["max_threads"] = min(
            1.0, (max_blocks_by_threads * block_threads) / s["max_threads"]
        )

        # Register limiter
        regs_per_block = block_threads * regs_per_thread
        if regs_per_block > 0:
            max_blocks_by_regs = s["max_regs"] // regs_per_block
            limiters["registers"] = min(
                1.0, (max_blocks_by_regs * block_threads) / s["max_threads"]
            )

        # Shared memory limiter
        if shared_mem > 0:
            max_blocks_by_smem = s["max_smem"] // shared_mem
            limiters["shared_memory"] = min(
                1.0, (max_blocks_by_smem * block_threads) / s["max_threads"]
            )

        return limiters

def _block_threads(features: KernelFeatureVector) -> int:
    x = features.block_size_x or 1
    y = features.block_size_y or 1
    z = features.block_size_z or 1
    return max(x * y * z, 1)
