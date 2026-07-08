"""AnalysisAgent — PTX analysis, feature extraction, and cost estimation.

Pipeline: extract PTX → parse → dependency graph → extract features → cost model.
This is the "enrichment" layer that adds depth to raw profiling metrics.
"""

from __future__ import annotations

from cuda_opt_agent.agents.base import AgentContext
from cuda_opt_agent.analysis.base import KernelAnalyzer
from cuda_opt_agent.build.nvcc_runner import NVCCRunner
from cuda_opt_agent.cost_models.base import CostModel
from cuda_opt_agent.features.pipeline import FeaturePipeline
from cuda_opt_agent.schemas.kernel import KernelSource

from cuda_opt_agent.core.logging import get_logger

log = get_logger(__name__)


class AnalysisAgent:
    """Run PTX analysis, feature extraction, and cost model prediction.

    Populates: ctx.ptx_source, ctx.analysis, ctx.features, ctx.cost_estimate.
    """

    def __init__(
        self,
        analyzer: KernelAnalyzer,
        feature_pipeline: FeaturePipeline,
        cost_model: CostModel,
        nvcc_runner: NVCCRunner | None = None,
    ) -> None:
        self._analyzer = analyzer
        self._features = feature_pipeline
        self._cost_model = cost_model
        self._nvcc = nvcc_runner

    async def run(self, ctx: AgentContext) -> AgentContext:
        """Execute analysis pipeline. Non-fatal — warnings on partial failure."""
        log.info("analysis.start", run_id=ctx.run_id)

        # ── Step 1: Extract PTX (if nvcc available) ────────────
        ptx_source = ctx.ptx_source  # May have been set by compile step
        if ptx_source is None and self._nvcc is not None:
            try:
                available = await self._nvcc.is_available()
                if available:
                    ptx_source = await self._nvcc.extract_ptx(ctx.kernel, arch=ctx.arch)
                    ctx.ptx_source = ptx_source
                    log.info("analysis.ptx_extracted", run_id=ctx.run_id, length=len(ptx_source))
            except Exception as exc:
                ctx.add_warning(f"PTX extraction failed: {exc}")
                log.warning("analysis.ptx_failed", run_id=ctx.run_id, error=str(exc))

        # ── Step 2: Run kernel analyzer ────────────────────────
        try:
            analyzer_available = await self._analyzer.is_available()
            if analyzer_available:
                analysis = await self._analyzer.analyze(
                    kernel=ctx.kernel,
                    ptx_source=ptx_source,
                    run_id=ctx.run_id,
                )
                ctx.analysis = analysis
                log.info(
                    "analysis.done",
                    run_id=ctx.run_id,
                    instructions=analysis.total_instructions,
                    ilp=round(analysis.ilp_estimate, 2),
                )
        except Exception as exc:
            ctx.add_warning(f"Kernel analysis failed: {exc}")
            log.warning("analysis.kernel_failed", run_id=ctx.run_id, error=str(exc))

        # ── Step 3: Extract features ───────────────────────────
        try:
            features = self._features.extract(
                kernel=ctx.kernel,
                metrics=ctx.profile_metrics,
                analysis=ctx.analysis,
            )
            ctx.features = features
            log.info(
                "analysis.features",
                run_id=ctx.run_id,
                available=features.available_count(),
            )
        except Exception as exc:
            ctx.add_warning(f"Feature extraction failed: {exc}")
            log.warning("analysis.features_failed", run_id=ctx.run_id, error=str(exc))

        # ── Step 4: Cost model prediction ──────────────────────
        if ctx.features is not None:
            try:
                estimate = self._cost_model.predict(
                    features=ctx.features,
                    metrics=ctx.profile_metrics,
                )
                ctx.cost_estimate = estimate
                log.info(
                    "analysis.cost_estimate",
                    run_id=ctx.run_id,
                    bound=estimate.bound_type.value,
                    confidence=round(estimate.confidence, 2),
                )
            except Exception as exc:
                ctx.add_warning(f"Cost model failed: {exc}")
                log.warning("analysis.cost_failed", run_id=ctx.run_id, error=str(exc))

        return ctx
