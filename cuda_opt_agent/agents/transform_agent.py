"""TransformAgent — LLM-advised transform suggestions.

Asks the LLM for optimization suggestions, then validates each
suggestion against available transforms. The LLM is ADVISORY only —
all suggestions are validated by deterministic code before use.
"""

from __future__ import annotations

from cuda_opt_agent.agents.base import AgentContext
from cuda_opt_agent.core.logging import get_logger
from cuda_opt_agent.llm.prompts.transform import build_transform_prompt
from cuda_opt_agent.llm.provider import LLMProvider
from cuda_opt_agent.schemas.transform import TransformSpec
from cuda_opt_agent.transforms.base import Transformation
from cuda_opt_agent.transforms.registry import get_transform, list_transforms

log = get_logger(__name__)


class TransformAgent:
    """Ask the LLM for transform suggestions and validate them.

    The LLM returns JSON with transform_name + params.
    This agent verifies each transform exists and is applicable.
    Invalid suggestions are silently dropped.
    """

    def __init__(self, llm_provider: LLMProvider) -> None:
        self._llm = llm_provider

    async def run(
        self,
        ctx: AgentContext,
        max_suggestions: int = 5,
    ) -> list[TransformSpec]:
        """Get LLM-suggested transforms for the current kernel.

        Args:
            ctx: Current pipeline context (kernel, bottleneck, metrics).
            max_suggestions: Maximum number of transforms to return.

        Returns:
            List of validated TransformSpec objects (may be empty).
        """
        if ctx.bottleneck is None:
            log.warning("transform_agent.no_bottleneck", run_id=ctx.run_id)
            return []

        log.info(
            "transform_agent.start",
            run_id=ctx.run_id,
            bottleneck=ctx.bottleneck.primary.value,
        )

        # Build prompt
        request = build_transform_prompt(
            kernel=ctx.kernel,
            bottleneck=ctx.bottleneck,
            metrics=ctx.profile_metrics,
        )

        # Ask LLM
        try:
            response = await self._llm.generate_json(request)
        except Exception as exc:
            log.error("transform_agent.llm_error", run_id=ctx.run_id, error=str(exc))
            return []

        # Validate response structure
        parsed = response.parsed_json
        if parsed is None:
            log.warning("transform_agent.no_json", run_id=ctx.run_id)
            return []

        if not isinstance(parsed, dict):
            log.warning("transform_agent.invalid_json_type", run_id=ctx.run_id)
            return []

        raw_transforms = parsed.get("transforms", [])
        if not isinstance(raw_transforms, list):
            log.warning("transform_agent.transforms_not_list", run_id=ctx.run_id)
            return []

        # Parse and validate suggestions
        valid_specs: list[TransformSpec] = []
        available = set(list_transforms())

        for raw in raw_transforms[:max_suggestions]:
            spec = self._validate_suggestion(raw, available, ctx)
            if spec is not None:
                valid_specs.append(spec)

        log.info(
            "transform_agent.done",
            run_id=ctx.run_id,
            raw_count=len(raw_transforms),
            valid_count=len(valid_specs),
            specs=[str(s) for s in valid_specs],
        )

        return valid_specs

    def _validate_suggestion(
        self,
        raw: dict,
        available: set[str],
        ctx: AgentContext,
    ) -> TransformSpec | None:
        """Validate a single LLM suggestion. Returns TransformSpec or None."""
        if not isinstance(raw, dict):
            log.debug("transform_agent.non_dict_entry")
            return None

        name = raw.get("transform_name", "")
        if not name or not isinstance(name, str):
            return None

        # Normalize name
        name = name.lower().strip().replace("-", "_").replace(" ", "_")

        if name not in available:
            log.debug("transform_agent.unknown_transform", name=name)
            return None

        try:
            transform = get_transform(name)
        except Exception:
            return None

        params = raw.get("params", {})
        if not isinstance(params, dict):
            params = {}

        # Check applicability
        if ctx.bottleneck is not None:
            try:
                if not transform.is_applicable(ctx.kernel, ctx.bottleneck, params):
                    log.debug("transform_agent.not_applicable", name=name)
                    return None
            except Exception:
                pass  # is_applicable may fail — still allow the attempt

        reason = raw.get("reason", "LLM suggestion")
        if not isinstance(reason, str):
            reason = "LLM suggestion"

        return TransformSpec(
            transform_name=name,
            params=params,
            reason=reason,
            source="llm",
        )
