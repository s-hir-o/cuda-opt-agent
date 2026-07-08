"""Shared memory tiling transform for canonical row-major matrix-multiply.

REQUIRES a valid MatmulProof from the structural parser.  If the proof
is missing or incomplete the transform rejects the kernel — it never
generates code from a guess.

Preserves proven buffer names, accumulation operator, and dimension variable.
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


@register_transform
class SharedMemoryTiling(Transformation):

    @property
    def name(self) -> str:
        return "shared_memory_tiling"

    @property
    def targets(self) -> list[BottleneckClass]:
        return [BottleneckClass.MEMORY_BOUND]

    @property
    def parameter_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "tile_size": {"type": "integer", "minimum": 1, "default": 16},
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
        proof = info.matmul_proof
        # Require a fully valid MatmulProof — no partial matches
        if proof is None or not proof.valid:
            return False
        # Additional safety: proven row-major layout
        if not proof.row_major_proven:
            return False
        # Must have proven dimension and accumulation operator
        if not proof.dim_var or not proof.accum_op:
            return False
        return True

    def apply(
        self,
        kernel: KernelSource | KernelVariant,
        params: dict[str, Any],
    ) -> TransformResult:
        tile = params.get("tile_size", 16)
        info = KernelParser.parse(kernel.source_code)
        proof = info.matmul_proof
        warnings: list[str] = []

        # Guard: should never happen if is_applicable was called
        if proof is None or not proof.valid:
            return self._make_result(
                kernel, kernel.source_code, params,
                "No valid matmul proof — cannot generate tiled code",
            )

        accum = proof.accumulation_var
        accum_op = proof.accum_op
        buf_a = proof.buf_a.name
        buf_b = proof.buf_b.name
        out_buf = proof.buf_out.name
        n_dim = proof.dim_var
        k_var = proof.k_loop.induction_var if proof.k_loop else "k"

        # Build accumulation statement from proven operator
        accum_stmt = f"{accum} {accum_op}="

        # Build tiled kernel body using ONLY proven symbol names
        tiled_body = f"""
    __shared__ float {buf_a}_s[{tile}][{tile}];
    __shared__ float {buf_b}_s[{tile}][{tile}];

    int row = blockIdx.y * {tile} + threadIdx.y;
    int col = blockIdx.x * {tile} + threadIdx.x;

    float {accum} = 0.0f;

    for (int t = 0; t < {n_dim}; t += {tile}) {{
        if (row < {n_dim} && (t + threadIdx.x) < {n_dim})
            {buf_a}_s[threadIdx.y][threadIdx.x] = {buf_a}[row * {n_dim} + t + threadIdx.x];
        else
            {buf_a}_s[threadIdx.y][threadIdx.x] = 0.0f;

        if ((t + threadIdx.y) < {n_dim} && col < {n_dim})
            {buf_b}_s[threadIdx.y][threadIdx.x] = {buf_b}[(t + threadIdx.y) * {n_dim} + col];
        else
            {buf_b}_s[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        for (int {k_var} = 0; {k_var} < {tile}; {k_var}++) {{
            {accum_stmt} {buf_a}_s[threadIdx.y][{k_var}] * {buf_b}_s[{k_var}][threadIdx.x];
        }}

        __syncthreads();
    }}

    if (row < {n_dim} && col < {n_dim})
        {out_buf}[row * {n_dim} + col] = {accum};
"""

        source = kernel.source_code
        sig_match = re.search(r"(__global__\s+void\s+\w+\s*\([^)]*)", source)
        if not sig_match:
            return self._make_result(kernel, source, params, "No kernel signature")
        brace_pos = source.find("{", sig_match.end())
        if brace_pos < 0:
            return self._make_result(kernel, source, params, "No opening brace")

        masked_full = mask_comments_and_strings(source)
        depth = 0
        close_pos = -1
        for i in range(brace_pos, len(source)):
            if not masked_full.is_code(i):
                continue
            if source[i] == "{":
                depth += 1
            elif source[i] == "}":
                depth -= 1
                if depth == 0:
                    close_pos = i
                    break
        if close_pos < 0:
            return self._make_result(kernel, source, params, "Unmatched braces")

        transformed = (
            source[: brace_pos + 1] + tiled_body + "\n}" + source[close_pos + 1 :]
        )

        warnings.append(
            f"Generated {tile}x{tile} tiling from MatmulProof: "
            f"buf_a={buf_a}, buf_b={buf_b}, out={out_buf}, "
            f"dim={n_dim}, accum_op={accum_op}"
        )
        result = self._make_result(
            kernel, transformed, params,
            f"{tile}x{tile} shared memory tiling (proven row-major)",
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
        if "__shared__" not in src:
            return False
        if "__syncthreads" not in src:
            return False
        # Verify all original pointer parameter names are preserved
        info_orig = KernelParser.parse(result.original_source)
        for p in info_orig.parameters:
            if p.is_pointer and p.name not in src:
                return False
        return True
