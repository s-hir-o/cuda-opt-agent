"""Schemas for LLM-generated CUDA kernel candidates."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from cuda_opt_agent.schemas.kernel import KernelSource, KernelVariant
from cuda_opt_agent.schemas.transform import TransformBackend, TransformSpec


class GeneratedKernelCandidate(BaseModel):
    """Complete CUDA source emitted by an LLM generator."""

    source_code: str
    entry_point: str
    rationale: str = ""
    assumptions: list[str] = Field(default_factory=list)
    launch_config: dict[str, Any] = Field(default_factory=dict)
    expected_improvement: float | None = None
    risks: list[str] = Field(default_factory=list)
    provider: str = ""
    model: str = ""

    def to_transform_spec(self) -> TransformSpec:
        """Represent this generated source in existing iteration metadata."""
        return TransformSpec(
            transform_name="llm_generated_kernel",
            backend=TransformBackend.TEXT,
            params={
                "entry_point": self.entry_point,
                "launch_config": self.launch_config,
                "expected_improvement": self.expected_improvement,
            },
            reason=self.rationale or "LLM generated a complete CUDA kernel",
            source="llm_generator",
        )

    def to_variant(
        self,
        parent: KernelSource,
        variant_id: str,
        iteration: int,
    ) -> KernelVariant:
        """Convert generated source to a normal KernelVariant."""
        return KernelVariant(
            variant_id=variant_id,
            parent_source=parent,
            source_code=self.source_code,
            language=parent.language,
            entry_point=self.entry_point,
            transform_history=[self.to_transform_spec()],
            iteration=iteration,
            metadata={
                "source": "llm_generator",
                "rationale": self.rationale,
                "assumptions": self.assumptions,
                "launch_config": self.launch_config,
                "expected_improvement": self.expected_improvement,
                "risks": self.risks,
                "provider": self.provider,
                "model": self.model,
            },
        )
