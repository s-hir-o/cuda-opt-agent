"""JSON schemas for structured LLM output.

Used by:
- generate_json() validation
- Prompt instructions ("respond in this format")
- OpenAI/Gemini JSON schema mode
"""

from __future__ import annotations
from typing import Any


TRANSFORM_RESPONSE_SCHEMA: dict[str, Any] = {
    "name": "transform_suggestions",
    "description": "Suggested CUDA kernel optimization transforms",
    "schema": {
        "type": "object",
        "properties": {
            "analysis_summary": {
                "type": "string",
                "description": "Brief analysis of the kernel bottlenecks",
            },
            "transforms": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "transform_name": {
                            "type": "string",
                            "description": "Transform identifier, e.g. shared_memory_tiling",
                        },
                        "params": {
                            "type": "object",
                            "description": "Transform-specific parameters",
                            "additionalProperties": True,
                        },
                        "reason": {
                            "type": "string",
                            "description": "Why this transform addresses the bottleneck",
                        },
                        "confidence": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 1,
                            "description": "Confidence that this transform will help",
                        },
                        "priority": {
                            "type": "integer",
                            "minimum": 1,
                            "description": "Execution order (1 = highest priority)",
                        },
                    },
                    "required": ["transform_name", "params", "reason", "confidence"],
                },
            },
            "warnings": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Potential risks or caveats",
            },
        },
        "required": ["analysis_summary", "transforms"],
    },
}


# Available transforms for prompt documentation
AVAILABLE_TRANSFORMS: list[dict[str, Any]] = [
    {
        "name": "shared_memory_tiling",
        "targets": ["memory_bound"],
        "params": {
            "tile_size": "int — tile dimension (e.g., 16, 32)",
        },
        "description": "Tile computation to use shared memory, reducing global memory traffic.",
    },
    {
        "name": "loop_unrolling",
        "targets": ["compute_bound", "latency_bound"],
        "params": {
            "factor": "int — unroll factor (2, 4, 8)",
        },
        "description": "Unroll inner loops to reduce branch overhead and enable ILP.",
    },
    {
        "name": "vectorized_loads",
        "targets": ["memory_bound"],
        "params": {
            "vector_width": "int — 2 or 4 (float2/float4)",
        },
        "description": "Use vector types for coalesced memory access.",
    },
    {
        "name": "register_blocking",
        "targets": ["compute_bound"],
        "params": {
            "block_m": "int",
            "block_n": "int",
        },
        "description": "Compute multiple output elements per thread to increase arithmetic intensity.",
    },
    {
        "name": "thread_coarsening",
        "targets": ["memory_bound", "latency_bound"],
        "params": {
            "factor": "int — elements per thread",
        },
        "description": "Each thread processes multiple elements, amortizing overhead.",
    },
    {
        "name": "warp_shuffle",
        "targets": ["memory_bound", "latency_bound"],
        "params": {
            "reduction_op": "string — 'sum', 'max', 'min'",
        },
        "description": "Replace shared-memory reductions with warp shuffle intrinsics.",
    },
    {
        "name": "kernel_fusion",
        "targets": ["memory_bound", "latency_bound"],
        "params": {
            "fused_kernel_name": "string — name of kernel to fuse with",
        },
        "description": "Fuse two kernels to eliminate intermediate global memory writes.",
    },
]
