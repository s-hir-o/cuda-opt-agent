
"""Roofline cost model.

Classifies a kernel as compute-bound, memory-bound, or mixed based on
throughput metrics and architectural limits.
"""

from __future__ import annotations

from cuda_opt_agent.cost_models.base import CostModel
from cuda_opt_agent.schemas.cost_model import BoundType, CostEstimate, ModelPrediction
from cuda_opt_agent.schemas.features import KernelFeatureVector
from cuda_opt_agent.schemas.metrics import ProfileMetrics

# Peak FP32 TFLOPS and memory bandwidth (GB/s) per architecture
_ARCH_SPECS: dict[str, dict[str, float]] = {
    "sm_80": {"peak_fp32_tflops": 19.5, "peak_bw_gbs": 2039},
    "sm_89": {"peak_fp32_tflops": 82.6, "peak_bw_gbs": 1008},
    "sm_90": {"peak_fp32_tflops": 67.0, "peak_bw_gbs": 3350},
}
_DEFAULT_ARCH = "sm_80"


class RooflineModel(CostModel):

    @property
    def name(self) -> str:
        return "roofline"

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
                bound_type=BoundType.UNKNOWN,
                confidence=0.0,
                reasoning="Insufficient data for roofline analysis",
            )

    def _predict_inner(
        self,
        features: KernelFeatureVector,
        metrics: ProfileMetrics | None,
    ) -> CostEstimate:
        # Determine bound type from throughput metrics
        sm_pct = features.sm_throughput_pct
        dram_pct = features.memory_throughput_pct

        if sm_pct is not None and dram_pct is not None:
            if dram_pct > sm_pct * 1.2:
                bound = BoundType.MEMORY
                confidence = min(dram_pct / 100.0, 1.0)
            elif sm_pct > dram_pct * 1.2:
                bound = BoundType.COMPUTE
                confidence = min(sm_pct / 100.0, 1.0)
            else:
                bound = BoundType.MIXED
                confidence = 0.5
        elif features.frac_memory_load is not None and features.frac_fma is not None:
            # Fallback: instruction mix heuristic
            mem_frac = (features.frac_memory_load or 0) + (features.frac_memory_store or 0)
            compute_frac = (features.frac_alu or 0) + (features.frac_fma or 0)
            if mem_frac > compute_frac * 1.5:
                bound = BoundType.MEMORY
                confidence = 0.4
            elif compute_frac > mem_frac * 1.5:
                bound = BoundType.COMPUTE
                confidence = 0.4
            else:
                bound = BoundType.MIXED
                confidence = 0.3
        else:
            bound = BoundType.UNKNOWN
            confidence = 0.1

        # Predicted time: use actual GPU time if available
        predicted_time = None
        if metrics and metrics.gpu_time_us is not None:
            predicted_time = metrics.gpu_time_us

        # Predicted occupancy from features
        predicted_occ = features.achieved_occupancy

        return CostEstimate(
            model_name=self.name,
            bound_type=bound,
            predicted_time_ms=predicted_time / 1000.0 if predicted_time else None,
            predicted_throughput_pct=max(
                sm_pct or 0, dram_pct or 0
            ) if (sm_pct or dram_pct) else None,
            predicted_occupancy=predicted_occ,
            predicted_register_usage=features.registers_per_thread,
            confidence=confidence,
            reasoning=_reasoning(bound, sm_pct, dram_pct),
        )

    def predict_speedup(
        self,
        baseline: KernelFeatureVector,
        variant: KernelFeatureVector,
    ) -> ModelPrediction:
        try:
            return self._predict_speedup_inner(baseline, variant)
        except Exception:
            return ModelPrediction(
                model_name=self.name,
                predicted_speedup=1.0,
                confidence=0.0,
                reasoning="Insufficient data for speedup prediction",
            )

    def _predict_speedup_inner(
        self,
        baseline: KernelFeatureVector,
        variant: KernelFeatureVector,
    ) -> ModelPrediction:
        improvements: list[str] = []
        risks: list[str] = []
        factors: list[float] = []

        # ILP improvement
        b_ilp = baseline.ilp_estimate
        v_ilp = variant.ilp_estimate
        if b_ilp and v_ilp and v_ilp > b_ilp:
            ratio = v_ilp / b_ilp
            factors.append(min(ratio, 2.0))
            improvements.append(f"ILP improved {b_ilp:.1f} → {v_ilp:.1f}")

        # Memory coalescing
        if baseline.is_coalesced is False and variant.is_coalesced is True:
            factors.append(2.0)
            improvements.append("Memory access coalesced")
        elif baseline.is_coalesced is False and variant.is_coalesced is False:
            risks.append("Still uncoalesced")

        # Occupancy
        b_occ = baseline.achieved_occupancy
        v_occ = variant.achieved_occupancy
        if b_occ and v_occ and v_occ > b_occ:
            ratio = v_occ / b_occ
            factors.append(min(ratio, 1.5))
            improvements.append(f"Occupancy {b_occ:.0%} → {v_occ:.0%}")

        # Register pressure
        b_reg = baseline.registers_per_thread
        v_reg = variant.registers_per_thread
        if b_reg and v_reg and v_reg > b_reg * 1.5:
            risks.append(f"Register pressure increased ({b_reg} → {v_reg})")
            factors.append(0.9)  # Slight pessimism

        # Bank conflicts
        if (variant.bank_conflict_count or 0) < (baseline.bank_conflict_count or 0):
            factors.append(1.2)
            improvements.append("Bank conflicts reduced")

        if not factors:
            speedup = 1.0
            conf = 0.1
        else:
            # Geometric mean of factors
            product = 1.0
            for f in factors:
                product *= f
            speedup = product ** (1.0 / len(factors))
            conf = min(0.3 * len(factors), 0.8)

        return ModelPrediction(
            model_name=self.name,
            predicted_speedup=round(speedup, 3),
            confidence=conf,
            reasoning=f"Estimated from {len(factors)} factors",
            improvement_areas=improvements,
            risk_areas=risks,
        )


def _reasoning(bound: BoundType, sm: float | None, dram: float | None) -> str:
    if bound == BoundType.MEMORY:
        return f"Memory-bound: DRAM {dram:.0f}% > SM {sm:.0f}%"
    if bound == BoundType.COMPUTE:
        return f"Compute-bound: SM {sm:.0f}% > DRAM {dram:.0f}%"
    if bound == BoundType.MIXED:
        return f"Mixed: SM {sm:.0f}%, DRAM {dram:.0f}%"
    return "Insufficient throughput data"
