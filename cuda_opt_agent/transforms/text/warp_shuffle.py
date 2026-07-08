"""Warp shuffle reduction transform.

REQUIRES a valid ReductionProof with single_warp_safe=True.  The reduction
loop is FULLY replaced — if any step of the rewrite cannot be completed,
the transform fails closed (returns success=False / not-applicable).

Partial, advisory-only, or comment-only rewrites are NEVER reported as
successful.
"""

from __future__ import annotations

import re
from typing import Any

from cuda_opt_agent.schemas.bottleneck import BottleneckClass, BottleneckResult
from cuda_opt_agent.schemas.kernel import KernelSource, KernelVariant
from cuda_opt_agent.schemas.transform import TransformResult
from cuda_opt_agent.transforms.analysis import KernelParser
from cuda_opt_agent.transforms.base import Transformation
from cuda_opt_agent.transforms.registry import register_transform
from cuda_opt_agent.transforms.text.lexical import mask_comments_and_strings
from cuda_opt_agent.transforms.validation import validate_transform

_COMBINE_TEMPLATES = {
    "+": "val + other",
    "*": "val * other",
    "max": "fmaxf(val, other)",
    "min": "fminf(val, other)",
}
_OP_DETECT_MAP = {
    "+": "+",
    "-": "+",  # conservative fallback
    "*": "*",
    "fmaxf": "max",
    "fminf": "min",
    "max": "max",
    "min": "min",
}


@register_transform
class WarpShuffleReduction(Transformation):

    @property
    def name(self) -> str:
        return "warp_shuffle"

    @property
    def targets(self) -> list[BottleneckClass]:
        return [BottleneckClass.MEMORY_BOUND, BottleneckClass.LATENCY_BOUND]

    @property
    def parameter_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "reduction_op": {
                    "type": "string",
                    "enum": ["sum", "max", "min"],
                    "default": "sum",
                },
            },
            "required": [],
        }

    def is_applicable(
        self,
        kernel: KernelSource | KernelVariant,
        bottleneck: BottleneckResult,
        params: dict[str, Any],
    ) -> bool:
        info = KernelParser.parse(kernel.source_code)
        proof = info.reduction_proof
        # Require a fully valid ReductionProof
        if proof is None or not proof.valid:
            return False
        # Must be proven single-warp safe
        if not proof.single_warp_safe:
            return False
        # Must have all required structural elements
        if not proof.halving_loop or not proof.halving_loop_var:
            return False
        if not proof.output_write_after_loop:
            return False
        if not proof.sequential_halving_proven:
            return False
        if not proof.syncthreads_in_halving_loop:
            return False
        return True

    def apply(
        self,
        kernel: KernelSource | KernelVariant,
        params: dict[str, Any],
    ) -> TransformResult:
        proof_info = KernelParser.parse(kernel.source_code)
        proof = proof_info.reduction_proof
        src = kernel.source_code
        warnings: list[str] = []

        # Fail closed: if proof is missing or incomplete, refuse to proceed
        if proof is None or not proof.valid:
            return self._make_result(
                kernel, src, params,
                "No valid reduction proof — cannot rewrite",
            )

        sbuf = proof.shared_buffer
        reduce_op_raw = proof.reduce_op
        halving_lp = proof.halving_loop
        lv = proof.halving_loop_var
        out_buf = proof.output_buffer

        # Map raw operator to our naming
        detected_op = _OP_DETECT_MAP.get(reduce_op_raw, "+")
        if params.get("reduction_op"):
            detected_op = params["reduction_op"]

        combine = _COMBINE_TEMPLATES.get(detected_op, "val + other")

        # Build the __device__ shuffle function
        shuffle_fn = f"""\
__device__ float warp_reduce_{detected_op}(float val) {{
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {{
        float other = __shfl_down_sync(0xFFFFFFFF, val, offset);
        val = {combine};
    }}
    return val;
}}
"""

        # Compute source-relative positions using body_start_offset
        loop_start_in_src = proof_info.body_start_offset + halving_lp.start_pos
        loop_end_in_src = proof_info.body_start_offset + halving_lp.end_pos

        # Validate computed positions
        if loop_start_in_src < 0 or loop_end_in_src > len(src):
            return self._make_result(
                kernel, src, params,
                "Cannot locate reduction loop in source — aborting",
            )

        # Verify the substring at the computed position looks like a for-loop
        snippet = src[loop_start_in_src:loop_start_in_src + 20].lstrip()
        if not snippet.startswith("for"):
            return self._make_result(
                kernel, src, params,
                "Computed loop position does not point to a for-loop — aborting",
            )

        new_reduction = f"""\
    // Warp-shuffle reduction (replaces shared-memory sequential halving)
    if (threadIdx.x < warpSize) {{
        float val = {sbuf}[threadIdx.x];
        val = warp_reduce_{detected_op}(val);
        {sbuf}[threadIdx.x] = val;
    }}
    __syncthreads();
"""

        # Insert shuffle function before kernel, replace reduction loop
        kernel_start = src.find("__global__")
        if kernel_start < 0:
            return self._make_result(kernel, src, params, "No kernel found — aborting")

        transformed = (
            src[:kernel_start]
            + shuffle_fn
            + "\n"
            + src[kernel_start:loop_start_in_src]
            + new_reduction
            + src[loop_end_in_src:]
        )

        # Verify the shuffle intrinsic is actually in the output
        if "__shfl_down_sync" not in transformed:
            return self._make_result(
                kernel, src, params,
                "Rewrite produced output without __shfl_down_sync — aborting",
            )

        # Verify the output write (if threadIdx == 0) is preserved
        if out_buf and out_buf not in transformed:
            return self._make_result(
                kernel, src, params,
                f"Output buffer '{out_buf}' missing from rewritten kernel — aborting",
            )

        warnings.append(
            f"Warp shuffle {detected_op} reduction: fully replaced sequential "
            f"halving loop with __shfl_down_sync. "
            f"Proven single-warp safe (halving initial < 32). "
            f"Only valid for blocks with <= 32 threads."
        )
        result = self._make_result(
            kernel, transformed, params,
            f"Warp shuffle {detected_op} reduction (complete rewrite)",
        )
        result.warnings = warnings
        # Promote success if validation passes
        if self.validate(result):
            result.success = True
        return result

    def validate(self, result: TransformResult) -> bool:
        if not validate_transform(
            result.transformed_source,
            original_source=result.original_source,
            result=result,
        ):
            return False
        src = result.transformed_source
        # Must contain the actual shuffle intrinsic in code (not comments)
        if "__shfl_down_sync" not in src:
            return False
        masked = mask_comments_and_strings(src)
        if not masked.filter_matches(re.compile(r"__shfl_down_sync")):
            return False
        # Must contain the device reduction function
        if "warp_reduce" not in src:
            return False
        # Original output buffer must still be present
        info_orig = KernelParser.parse(result.original_source)
        if info_orig.output_write_buffer:
            if info_orig.output_write_buffer not in src:
                return False
        return True
