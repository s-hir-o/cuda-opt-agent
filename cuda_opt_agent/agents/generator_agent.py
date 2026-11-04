"""GeneratorAgent - LLM-driven complete CUDA source generation.

The agent only creates candidate source. Compilation, correctness, and
benchmarking remain owned by BenchmarkAgent and the existing coordinator path.
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import ValidationError as PydanticValidationError

from cuda_opt_agent.agents.base import AgentContext
from cuda_opt_agent.core.logging import get_logger
from cuda_opt_agent.llm.prompts.generation import build_generation_prompt
from cuda_opt_agent.llm.provider import LLMProvider
from cuda_opt_agent.schemas.generation import GeneratedKernelCandidate

log = get_logger(__name__)


class GeneratorAgent:
    """Ask an LLM for complete CUDA kernel candidates."""

    def __init__(self, llm_provider: LLMProvider) -> None:
        self._llm = llm_provider

    async def run(
        self,
        ctx: AgentContext,
        max_candidates: int = 1,
        required_entry_point: str | None = None,
        require_runnable_harness: bool = False,
    ) -> list[GeneratedKernelCandidate]:
        """Generate and validate CUDA source candidates."""
        log.info("generator_agent.start", run_id=ctx.run_id)

        request = build_generation_prompt(
            kernel=ctx.kernel,
            bottleneck=ctx.bottleneck,
            metrics=ctx.profile_metrics,
            analysis=ctx.analysis,
            features=ctx.features,
            cost_estimate=ctx.cost_estimate,
            max_candidates=max_candidates,
            required_entry_point=required_entry_point,
            require_runnable_harness=require_runnable_harness,
        )

        try:
            response = await self._llm.generate_json(request)
        except Exception as exc:
            log.error("generator_agent.llm_error", run_id=ctx.run_id, error=str(exc))
            return []

        parsed = response.parsed_json
        raw_candidates = _extract_candidates(parsed)
        if not raw_candidates:
            log.warning("generator_agent.no_candidates", run_id=ctx.run_id)
            return []

        candidates: list[GeneratedKernelCandidate] = []
        for raw in raw_candidates[:max_candidates]:
            candidate = self._parse_candidate(
                raw,
                response.provider,
                response.model,
                required_entry_point=required_entry_point,
                require_runnable_harness=require_runnable_harness,
            )
            if candidate is not None:
                candidates.append(candidate)

        log.info(
            "generator_agent.done",
            run_id=ctx.run_id,
            raw_count=len(raw_candidates),
            valid_count=len(candidates),
        )
        return candidates

    def _parse_candidate(
        self,
        raw: dict[str, Any],
        provider: str,
        model: str,
        required_entry_point: str | None = None,
        require_runnable_harness: bool = False,
    ) -> GeneratedKernelCandidate | None:
        if not isinstance(raw, dict):
            return None

        try:
            candidate = GeneratedKernelCandidate.model_validate(raw)
        except PydanticValidationError as exc:
            log.debug("generator_agent.schema_invalid", error=str(exc))
            return None

        candidate.provider = candidate.provider or provider
        candidate.model = candidate.model or model

        if required_entry_point and candidate.entry_point != required_entry_point:
            log.debug(
                "generator_agent.entry_point_changed",
                required=required_entry_point,
                candidate=candidate.entry_point,
            )
            return None

        if not _entry_point_declared(candidate.source_code, candidate.entry_point):
            log.debug("generator_agent.entry_point_missing", entry_point=candidate.entry_point)
            return None

        if not _structurally_valid_source(candidate.source_code):
            log.debug("generator_agent.structural_validation_failed")
            return None

        if require_runnable_harness and not _has_runnable_harness(candidate.source_code):
            log.debug("generator_agent.runnable_harness_missing")
            return None

        return candidate


def _extract_candidates(parsed: Any) -> list[dict[str, Any]]:
    """Extract candidate dictionaries from supported JSON response shapes."""
    if isinstance(parsed, dict):
        candidates = parsed.get("candidates")
        if isinstance(candidates, list):
            return [c for c in candidates if isinstance(c, dict)]
        if "source_code" in parsed:
            return [parsed]
    return []


def _entry_point_declared(source: str, entry_point: str) -> bool:
    """Return True if source declares the requested global kernel."""
    if not entry_point:
        return False
    pattern = re.compile(
        r"__global__\s+void\s+" + re.escape(entry_point) + r"\s*\(",
        re.MULTILINE,
    )
    return pattern.search(source) is not None


def _structurally_valid_source(source: str) -> bool:
    """Lightweight generated-source guard before nvcc/correctness gates."""
    if "__global__" not in source or "void" not in source:
        return False
    if "TODO" in source or "placeholder" in source:
        return False
    return (
        _balanced(source, "{", "}")
        and _balanced(source, "(", ")")
        and _balanced(source, "[", "]")
    )


def _has_runnable_harness(source: str) -> bool:
    """Return True when generated source keeps the executable correctness contract."""
    has_main = re.search(r"\bint\s+main\s*\(", source) is not None
    has_output_flag = '"--output"' in source or "'--output'" in source
    return has_main and has_output_flag


def _balanced(source: str, open_ch: str, close_ch: str) -> bool:
    depth = 0
    for ch in source:
        if ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth < 0:
                return False
    return depth == 0
