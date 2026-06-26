"""Extract features from KernelAnalysis (PTX-level)."""

from __future__ import annotations

from cuda_opt_agent.features.base import FeatureExtractor
from cuda_opt_agent.schemas.analysis import KernelAnalysis
from cuda_opt_agent.schemas.features import KernelFeatureVector
from cuda_opt_agent.schemas.kernel import KernelSource, KernelVariant
from cuda_opt_agent.schemas.metrics import ProfileMetrics


class PTXFeatureExtractor(FeatureExtractor):

    @property
    def name(self) -> str:
        return "ptx"

    @property
    def required_inputs(self) -> list[str]:
        return ["analysis"]

    def extract(
        self,
        kernel: KernelSource | KernelVariant,
        metrics: ProfileMetrics | None,
        analysis: KernelAnalysis | None,
    ) -> KernelFeatureVector:
        if analysis is None:
            return KernelFeatureVector()

        total = analysis.total_instructions or 1  # avoid division by zero
        mix = analysis.instruction_mix

        return KernelFeatureVector(
            # Instruction mix
            frac_alu=_frac(mix.get("alu", 0), total),
            frac_fma=_frac(mix.get("fma", 0), total),
            frac_memory_load=_frac(
                mix.get("memory_load", 0) + mix.get("shared_load", 0), total
            ),
            frac_memory_store=_frac(
                mix.get("memory_store", 0) + mix.get("shared_store", 0), total
            ),
            frac_shared=_frac(
                mix.get("shared_load", 0) + mix.get("shared_store", 0), total
            ),
            frac_control_flow=_frac(mix.get("control_flow", 0), total),
            # Parallelism
            ilp_estimate=analysis.ilp_estimate,
            critical_path_length=analysis.critical_path_length,
            total_instructions=analysis.total_instructions,
            # Memory
            is_coalesced=not analysis.has_uncoalesced_access,
            has_shared_memory=analysis.has_shared_memory,
            bank_conflict_count=analysis.bank_conflict_count or 0,
            # Compute
            has_divergent_branches=analysis.has_divergent_branches,
            loop_nest_depth=analysis.loop_nest_depth or None,
        )


def _frac(count: int, total: int) -> float:
    return count / total if total > 0 else 0.0
