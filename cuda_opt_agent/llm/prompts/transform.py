"""Prompt builder for transform suggestions.

The core prompt used by TransformAgent: given a kernel, its bottleneck
classification, and profiling metrics, ask the LLM to suggest transforms.

The LLM is ADVISORY only — all suggestions are validated by deterministic code
before application.
"""

from __future__ import annotations

from cuda_opt_agent.llm.prompts.base import (
    format_bottleneck,
    format_kernel_source,
    format_metrics,
    format_variant_diff,
)
from cuda_opt_agent.llm.prompts.schemas import AVAILABLE_TRANSFORMS, TRANSFORM_RESPONSE_SCHEMA
from cuda_opt_agent.schemas.bottleneck import BottleneckResult
from cuda_opt_agent.schemas.kernel import KernelSource, KernelVariant
from cuda_opt_agent.schemas.llm import LLMRequest
from cuda_opt_agent.schemas.metrics import ProfileMetrics

import json


_TRANSFORM_LIST_TEXT = "\n".join(
    f"- **{t['name']}** (targets: {', '.join(t['targets'])}): {t['description']}\n"
    f"  Parameters: `{json.dumps(t['params'])}`"
    for t in AVAILABLE_TRANSFORMS
)

SYSTEM_PROMPT = """\
You are an expert CUDA optimization engineer. Your task is to analyze a GPU kernel \
and suggest specific optimization transforms.

## Available Transforms

{_transform_list}

## Rules

1. Suggest ONLY transforms from the list above.
2. Each suggestion must include concrete parameter values, not ranges.
3. Prioritize transforms that directly address the identified bottleneck.
4. Consider the transform history — avoid repeating applied transforms unless \
with different parameters.
5. If the kernel is already well-optimized, say so and suggest no transforms.
6. Respond in the exact JSON format specified.

## Response Format

```json
{schema}
```""".format(
    _transform_list=_TRANSFORM_LIST_TEXT,
    schema=json.dumps(TRANSFORM_RESPONSE_SCHEMA["schema"], indent=2),
)


def build_transform_prompt(
    kernel: KernelSource | KernelVariant,
    bottleneck: BottleneckResult,
    metrics: ProfileMetrics | None = None,
) -> LLMRequest:
    """Build an LLM request for transform suggestions.

    Args:
        kernel: The kernel to optimize (source or variant).
        bottleneck: Bottleneck classification from the classifier agent.
        metrics: Optional profiling metrics for richer context.

    Returns:
        Ready-to-send LLMRequest with system + user prompts.
    """
    sections: list[str] = [
        "# Kernel to Optimize\n",
        format_kernel_source(kernel),
        "\n\n# Bottleneck Analysis\n",
        format_bottleneck(bottleneck),
    ]

    if metrics is not None:
        sections.extend(["\n\n", format_metrics(metrics)])

    if isinstance(kernel, KernelVariant) and kernel.transform_history:
        sections.extend(["\n\n", format_variant_diff(kernel)])

    sections.append(
        "\n\n# Task\n"
        "Suggest optimization transforms for this kernel. "
        "Return your response as JSON matching the specified schema."
    )

    return LLMRequest(
        prompt="".join(sections),
        system_prompt=SYSTEM_PROMPT,
        json_mode=True,
    )
