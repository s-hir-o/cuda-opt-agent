"""Prompt builder for complete CUDA kernel generation."""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel

from cuda_opt_agent.llm.prompts.base import (
    format_bottleneck,
    format_kernel_source,
    format_metrics,
)
from cuda_opt_agent.schemas.bottleneck import BottleneckResult
from cuda_opt_agent.schemas.kernel import KernelSource, KernelVariant
from cuda_opt_agent.schemas.llm import LLMRequest
from cuda_opt_agent.schemas.metrics import ProfileMetrics


GENERATION_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["candidates"],
    "properties": {
        "candidates": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["source_code", "entry_point"],
                "properties": {
                    "source_code": {
                        "type": "string",
                        "description": (
                            "Complete CUDA source code for the candidate, including "
                            "the host harness when present in the input."
                        ),
                    },
                    "entry_point": {
                        "type": "string",
                        "description": "Name of the generated __global__ kernel.",
                    },
                    "rationale": {"type": "string"},
                    "assumptions": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "launch_config": {"type": "object"},
                    "expected_improvement": {"type": ["number", "null"]},
                    "risks": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
            },
        }
    },
}


SYSTEM_PROMPT = """\
You are a CUDA kernel code generator.

Your only job is to generate complete CUDA source candidates. Do not profile,
benchmark, or decide correctness. The caller will compile, validate correctness,
and benchmark every candidate.

Rules:
1. Return only JSON matching the schema.
2. Each candidate must include complete CUDA source code, not a patch or diff.
3. Preserve the original kernel's semantic intent and input/output contract.
4. Keep the generated source self-contained and compilable by nvcc.
5. Preserve the original host-side harness and correctness/output contract.
6. Do not invent new host-side benchmarking or correctness behavior.
7. Avoid placeholder symbols and TODOs.
8. Prefer one high-quality candidate over many weak candidates.

Response schema:
```json
{schema}
```""".format(schema=json.dumps(GENERATION_RESPONSE_SCHEMA, indent=2))


def build_generation_prompt(
    kernel: KernelSource | KernelVariant,
    bottleneck: BottleneckResult | None = None,
    metrics: ProfileMetrics | None = None,
    analysis: BaseModel | None = None,
    features: BaseModel | None = None,
    cost_estimate: BaseModel | None = None,
    max_candidates: int = 1,
    required_entry_point: str | None = None,
    require_runnable_harness: bool = False,
) -> LLMRequest:
    """Build an LLM request for complete CUDA source generation."""
    sections: list[str] = [
        "# Kernel To Improve\n",
        format_kernel_source(kernel),
    ]

    if bottleneck is not None:
        sections.extend(["\n\n# Bottleneck Analysis\n", format_bottleneck(bottleneck)])

    if metrics is not None:
        sections.extend(["\n\n", format_metrics(metrics)])

    if analysis is not None:
        sections.extend(["\n\n# Static/PTX Analysis\n", _format_model(analysis)])

    if features is not None:
        sections.extend(["\n\n# Feature Vector\n", _format_model(features)])

    if cost_estimate is not None:
        sections.extend(["\n\n# Cost Estimate\n", _format_model(cost_estimate)])

    sections.append(
        "\n\n# Task\n"
        f"Generate up to {max_candidates} complete CUDA kernel candidate(s). "
        "Return JSON only. The generated source will be structurally validated, "
        "compiled with nvcc, checked for correctness, and benchmarked by the "
        "existing pipeline."
    )
    if required_entry_point:
        sections.append(
            "\n\nThe generated CUDA source must preserve the original kernel "
            f"entry point exactly: `{required_entry_point}`. Set each "
            "candidate's `entry_point` field to that exact value and declare "
            f"`__global__ void {required_entry_point}(...)` in the source."
        )
    if require_runnable_harness:
        sections.append(
            "\n\nThe generated source must remain a complete runnable CUDA "
            "translation unit. Preserve the host `int main(...)` harness, keep "
            "support for `--output <path>`, and write the same raw numeric "
            "output buffer format as the original source. Optimize the CUDA "
            "kernel implementation without removing the executable harness."
        )

    return LLMRequest(
        prompt="".join(sections),
        system_prompt=SYSTEM_PROMPT,
        json_mode=True,
    )


def _format_model(model: BaseModel) -> str:
    """Format a Pydantic model compactly for prompt context."""
    return "```json\n" + model.model_dump_json(indent=2) + "\n```"
