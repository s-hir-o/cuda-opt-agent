"""ProfilerAgent — runs GPU profiling on a compiled kernel.

Wraps the Profiler interface with availability checking and metrics validation.
"""

from __future__ import annotations

from cuda_opt_agent.agents.base import AgentContext
from cuda_opt_agent.core.logging import get_logger
from cuda_opt_agent.profiling.metrics_validator import validate_metrics
from cuda_opt_agent.profiling.profiler import Profiler

log = get_logger(__name__)


class ProfilerAgent:
    """Profile a kernel and populate AgentContext with ProfileMetrics."""

    def __init__(self, profiler: Profiler) -> None:
        self._profiler = profiler

    async def run(self, ctx: AgentContext) -> AgentContext:
        """Execute profiling. Populates ctx.profile_metrics.

        On failure: adds error to ctx, leaves profile_metrics as None.
        """
        log.info("profiler.start", run_id=ctx.run_id, kernel=_kernel_name(ctx))

        available = await self._profiler.is_available()
        if not available:
            msg = "Profiler not available — skipping profiling"
            log.warning("profiler.unavailable", run_id=ctx.run_id)
            ctx.add_warning(msg)
            return ctx

        try:
            metrics = await self._profiler.profile(
                kernel=ctx.kernel,
                run_id=ctx.run_id,
                arch=ctx.arch,
            )

            # Validate
            warnings = validate_metrics(metrics)
            for w in warnings:
                ctx.add_warning(f"Metrics: {w}")
                log.warning("profiler.metric_warning", warning=w)

            ctx.profile_metrics = metrics
            log.info(
                "profiler.done",
                run_id=ctx.run_id,
                gpu_time_us=metrics.gpu_time_us,
            )

        except Exception as exc:
            msg = f"Profiling failed: {exc}"
            ctx.add_error(msg)
            log.error("profiler.error", run_id=ctx.run_id, error=str(exc))

        return ctx


def _kernel_name(ctx: AgentContext) -> str:
    from cuda_opt_agent.schemas.kernel import KernelSource
    if isinstance(ctx.kernel, KernelSource):
        return ctx.kernel.name
    return ctx.kernel.variant_id
