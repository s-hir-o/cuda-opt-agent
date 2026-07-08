"""Register blocking transform for proven canonical row-major matmul.

REQUIRES a valid MatmulProof.  Each thread computes a BM x BN tile of
output elements using proven buffer names, accumulation operator, and
dimension variable.  Rejects anything the parser cannot fully verify.
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
class RegisterBlocking(Transformation):

    @property
    def name(self) -> str:
        return "register_blocking"

    @property
    def targets(self) -> list[BottleneckClass]:
        return [BottleneckClass.COMPUTE_BOUND]

    @property
    def parameter_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "block_m": {"type": "integer", "minimum": 1, "default": 2},
                "block_n": {"type": "integer", "minimum": 1, "default": 2},
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
        # Require a fully valid MatmulProof
        if proof is None or not proof.valid:
            return False
        if not proof.row_major_proven:
            return False
        if not proof.dim_var or not proof.accum_op:
            return False
        # Reject if shared memory already present (already tiled)
        if info.has_shared_memory:
            return False
        return True

    def apply(
        self,
        kernel: KernelSource | KernelVariant,
        params: dict[str, Any],
    ) -> TransformResult:
        bm = params.get("block_m", 2)
        bn = params.get("block_n", 2)
        info = KernelParser.parse(kernel.source_code)
        proof = info.matmul_proof
        warnings: list[str] = []

        if proof is None or not proof.valid:
            return self._make_result(
                kernel, kernel.source_code, params,
                "No valid matmul proof — cannot generate register-blocked code",
            )

        accum = proof.accumulation_var
        accum_op = proof.accum_op
        buf_a = proof.buf_a.name
        buf_b = proof.buf_b.name
        out_buf = proof.buf_out.name
        n_param = proof.dim_var
        k_var = proof.k_loop.induction_var if proof.k_loop else "k"

        # Build accumulator declarations
        accum_decl = "\n".join(
            f"    float {accum}_{i}_{j} = 0.0f;"
            for i in range(bm)
            for j in range(bn)
        )

        # Build compute block using proven accumulation operator
        compute_lines = []
        for i in range(bm):
            for j in range(bn):
                compute_lines.append(
                    f"        {accum}_{i}_{j} {accum_op}= "
                    f"{buf_a}[(row + {i}) * {n_param} + {k_var}] "
                    f"* {buf_b}[{k_var} * {n_param} + (col + {j})];"
                )
        compute_block = "\n".join(compute_lines)

        # Build store block
        store_lines = []
        for i in range(bm):
            for j in range(bn):
                store_lines.append(
                    f"        {out_buf}[(row + {i}) * {n_param} + (col + {j})] = {accum}_{i}_{j};"
                )
        store_block = "\n".join(store_lines)

        source = kernel.source_code
        sig_match = re.search(r"(__global__\s+void\s+\w+\s*\([^)]*\))", source)
        if not sig_match:
            return self._make_result(kernel, source, params, "No kernel signature")
        brace_pos = source.find("{", sig_match.end())

        masked = mask_comments_and_strings(source)
        depth = 0
        close_pos = -1
        for i in range(brace_pos, len(source)):
            if not masked.is_code(i):
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

        new_body = f"""
    int row = blockIdx.y * blockDim.y * {bm} + threadIdx.y * {bm};
    int col = blockIdx.x * blockDim.x * {bn} + threadIdx.x * {bn};

{accum_decl}

    for (int {k_var} = 0; {k_var} < {n_param}; {k_var}++) {{
{compute_block}
    }}

    if (row + {bm - 1} < {n_param} && col + {bn - 1} < {n_param}) {{
{store_block}
    }}
"""
        transformed = (
            source[: brace_pos + 1] + new_body + "\n}" + source[close_pos + 1 :]
        )

        warnings.append(
            f"Register blocking {bm}x{bn} from MatmulProof: "
            f"buf_a={buf_a}, buf_b={buf_b}, out={out_buf}, "
            f"dim={n_param}, accum_op={accum_op}. "
            f"Each thread computes {bm * bn} outputs."
        )
        result = self._make_result(
            kernel, transformed, params,
            f"Register blocking {bm}x{bn} (proven row-major)",
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
            check_comments=False,
        ):
            return False
        src = result.transformed_source
        # Must have accumulators with the _i_j pattern
        if not re.search(r"\w+_\d+_\d+", src):
            return False
        # Output buffer from original must be present
        info_orig = KernelParser.parse(result.original_source)
        if info_orig.output_write_buffer and info_orig.output_write_buffer not in src:
            return False
        return True
