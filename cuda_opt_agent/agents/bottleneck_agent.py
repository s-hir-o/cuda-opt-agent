"""BottleneckAgent — deterministic bottleneck classification.

Uses profiling metrics, features, and cost model to classify the
primary bottleneck. Purely rule-based — no LLM involvement.

Classification rules (in priority order):
1. Occupancy limited → OCCUPANCY_BOUND
2. Memory throughput dominant → MEMORY_BOUND
3. Compute throughput dominant → COMPUTE_BOUND
4. Divergent branches detected → DIVERGENCE_BOUND
5. Low ILP with moderate throughput → LATENCY_BOUND
6. Default → UNCLASSIFIED
"""

from __future__ import annotations

from cuda_opt_agent.agents.base import AgentContext
from cuda_opt_agent.core.logging import get_logger
from cuda_opt_agent.schemas.bottleneck import BottleneckClass, BottleneckResult
from cuda_opt_agent.schemas.cost_model import BoundType
from cuda_opt_agent.transforms.registry import transforms_for_bottleneck

log = get_logger(__name__)

# Thresholds
_DRAM_DOMINANT_RATIO = 1.3  # DRAM > SM * 1.3 → memory bound
_SM_DOMINANT_RATIO = 1.3    # SM > DRAM * 1.3 → compute bound
_HIGH_THROUGHPUT = 60.0     # % of peak
_LOW_OCCUPANCY = 0.5        # fraction
_LOW_ILP = 1.5


class BottleneckAgent:
    """Classify kernel bottleneck using deterministic rules."""

    def run(self, ctx: AgentContext) -> AgentContext:
        """Classify bottleneck. Populates ctx.bottleneck.

        Uses metrics, features, and cost model (whatever is available).
        Never raises — returns UNCLASSIFIED on insufficient data.
        """
        log.info("bottleneck.start", run_id=ctx.run_id)

        result = self._classify(ctx)
        ctx.bottleneck = result

        log.info(
            "bottleneck.done",
            run_id=ctx.run_id,
            primary=result.primary.value,
            confidence=round(result.confidence, 2),
            recommended=result.recommended_transforms,
        )
        return ctx

    def _classify(self, ctx: AgentContext) -> BottleneckResult:
        metrics = ctx.profile_metrics
        features = ctx.features
        cost = ctx.cost_estimate
        evidence: dict = {}
        reasoning_parts: list[str] = []

        # ── Gather evidence ────────────────────────────────────
        dram = metrics.dram_throughput if metrics else None
        sm = metrics.sm_throughput if metrics else None
        occ = metrics.achieved_occupancy if metrics else None
        regs = metrics.registers_per_thread if metrics else None
        reg_limited = metrics.occupancy_limited_by_registers if metrics else None
        smem_limited = metrics.occupancy_limited_by_shared_mem if metrics else None

        ilp = features.ilp_estimate if features else None
        divergent = features.has_divergent_branches if features else None
        bank_conflicts = features.bank_conflict_count if features else None

        cost_bound = cost.bound_type if cost else None
        cost_conf = cost.confidence if cost else 0.0

        evidence = {
            "dram_throughput": dram,
            "sm_throughput": sm,
            "achieved_occupancy": occ,
            "registers_per_thread": regs,
            "occupancy_limited_by_registers": reg_limited,
            "ilp_estimate": ilp,
            "has_divergent_branches": divergent,
            "bank_conflict_count": bank_conflicts,
            "cost_model_bound": cost_bound.value if cost_bound else None,
        }

        # ── Rule 1: Occupancy limited ─────────────────────────
        if reg_limited and occ is not None and occ < _LOW_OCCUPANCY:
            reasoning_parts.append(
                f"Occupancy {occ:.0%} limited by registers ({regs} regs/thread)"
            )
            return BottleneckResult(
                primary=BottleneckClass.OCCUPANCY_BOUND,
                confidence=0.85,
                reasoning="; ".join(reasoning_parts),
                evidence=evidence,
                recommended_transforms=_transforms_for(BottleneckClass.OCCUPANCY_BOUND),
            )

        if smem_limited and occ is not None and occ < _LOW_OCCUPANCY:
            reasoning_parts.append(f"Occupancy {occ:.0%} limited by shared memory")
            return BottleneckResult(
                primary=BottleneckClass.OCCUPANCY_BOUND,
                confidence=0.80,
                reasoning="; ".join(reasoning_parts),
                evidence=evidence,
                recommended_transforms=_transforms_for(BottleneckClass.OCCUPANCY_BOUND),
            )

        # ── Rule 2: Memory bound ──────────────────────────────
        if dram is not None and sm is not None:
            if dram > sm * _DRAM_DOMINANT_RATIO and dram > _HIGH_THROUGHPUT * 0.5:
                reasoning_parts.append(
                    f"DRAM throughput {dram:.0f}% >> SM throughput {sm:.0f}%"
                )
                conf = min(dram / 100.0, 0.95)
                # Boost confidence if cost model agrees
                if cost_bound == BoundType.MEMORY:
                    conf = min(conf + 0.1 * cost_conf, 0.95)
                    reasoning_parts.append("Cost model confirms memory-bound")
                if bank_conflicts and bank_conflicts > 0:
                    reasoning_parts.append(f"{bank_conflicts} bank conflicts detected")
                return BottleneckResult(
                    primary=BottleneckClass.MEMORY_BOUND,
                    confidence=conf,
                    reasoning="; ".join(reasoning_parts),
                    evidence=evidence,
                    recommended_transforms=_transforms_for(BottleneckClass.MEMORY_BOUND),
                )

        # ── Rule 3: Compute bound ─────────────────────────────
        if sm is not None and dram is not None:
            if sm > dram * _SM_DOMINANT_RATIO and sm > _HIGH_THROUGHPUT * 0.5:
                reasoning_parts.append(
                    f"SM throughput {sm:.0f}% >> DRAM throughput {dram:.0f}%"
                )
                conf = min(sm / 100.0, 0.95)
                if cost_bound == BoundType.COMPUTE:
                    conf = min(conf + 0.1 * cost_conf, 0.95)
                    reasoning_parts.append("Cost model confirms compute-bound")
                return BottleneckResult(
                    primary=BottleneckClass.COMPUTE_BOUND,
                    confidence=conf,
                    reasoning="; ".join(reasoning_parts),
                    evidence=evidence,
                    recommended_transforms=_transforms_for(BottleneckClass.COMPUTE_BOUND),
                )

        # ── Rule 4: Divergence bound ──────────────────────────
        if divergent:
            reasoning_parts.append("Divergent branches detected in kernel")
            return BottleneckResult(
                primary=BottleneckClass.DIVERGENCE_BOUND,
                confidence=0.70,
                reasoning="; ".join(reasoning_parts),
                evidence=evidence,
                recommended_transforms=_transforms_for(BottleneckClass.DIVERGENCE_BOUND),
            )

        # ── Rule 5: Latency bound ─────────────────────────────
        if ilp is not None and ilp < _LOW_ILP:
            reasoning_parts.append(f"Low ILP ({ilp:.1f}) — latency-bound execution")
            return BottleneckResult(
                primary=BottleneckClass.LATENCY_BOUND,
                confidence=0.55,
                reasoning="; ".join(reasoning_parts),
                evidence=evidence,
                recommended_transforms=_transforms_for(BottleneckClass.LATENCY_BOUND),
            )

        # ── Rule 6: Use cost model if available ───────────────
        if cost_bound and cost_bound != BoundType.UNKNOWN:
            reasoning_parts.append(f"Cost model classifies as {cost_bound.value}")
            cls = {
                BoundType.MEMORY: BottleneckClass.MEMORY_BOUND,
                BoundType.COMPUTE: BottleneckClass.COMPUTE_BOUND,
                BoundType.LATENCY: BottleneckClass.LATENCY_BOUND,
                BoundType.MIXED: BottleneckClass.LATENCY_BOUND,
            }.get(cost_bound, BottleneckClass.UNCLASSIFIED)
            return BottleneckResult(
                primary=cls,
                confidence=max(cost_conf, 0.3),
                reasoning="; ".join(reasoning_parts),
                evidence=evidence,
                recommended_transforms=_transforms_for(cls),
            )

        # ── Fallback ──────────────────────────────────────────
        reasoning_parts.append("Insufficient data for classification")
        return BottleneckResult(
            primary=BottleneckClass.UNCLASSIFIED,
            confidence=0.1,
            reasoning="; ".join(reasoning_parts),
            evidence=evidence,
            recommended_transforms=[],
        )


def _transforms_for(cls: BottleneckClass) -> list[str]:
    """Return names of registered transforms that target this bottleneck."""
    try:
        return [t.name for t in transforms_for_bottleneck(cls)]
    except Exception:
        return []
