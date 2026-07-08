"""Thread coarsening transform.

REQUIRES a valid CoarseningProof from the parser.  Only applies to
simple single-index kernels with a proven safe pattern.
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
class ThreadCoarsening(Transformation):

    @property
    def name(self) -> str:
        return "thread_coarsening"

    @property
    def targets(self) -> list[BottleneckClass]:
        return [BottleneckClass.MEMORY_BOUND, BottleneckClass.LATENCY_BOUND]

    @property
    def parameter_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "factor": {
                    "type": "integer",
                    "minimum": 2,
                    "maximum": 16,
                    "default": 4,
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
        proof = info.coarsening_proof
        # Require a fully valid CoarseningProof
        if proof is None or not proof.valid:
            return False
        return True

    def apply(
        self,
        kernel: KernelSource | KernelVariant,
        params: dict[str, Any],
    ) -> TransformResult:
        factor = params.get("factor", 4)
        info = KernelParser.parse(kernel.source_code)
        proof = info.coarsening_proof
        src = kernel.source_code
        masked = mask_comments_and_strings(src)
        warnings: list[str] = []

        if proof is None or not proof.valid:
            return self._make_result(
                kernel, src, params,
                "No valid coarsening proof — cannot coarsen",
            )

        var_name = proof.idx_var
        dim = proof.idx_dim

        coarsened = (
            f"for (int {var_name} = blockIdx.{dim} * blockDim.{dim} + threadIdx.{dim}; "
            f"{var_name} < N; "
            f"{var_name} += gridDim.{dim} * blockDim.{dim} * {factor})"
        )

        # Find the index computation line in source
        idx_pat = re.compile(
            rf"int\s+{re.escape(var_name)}\s*=\s*blockIdx\.{re.escape(dim)}"
            rf"\s*\*\s*blockDim\.{re.escape(dim)}\s*\+\s*threadIdx\.{re.escape(dim)}\s*;"
        )
        idx_matches = masked.filter_matches(idx_pat)
        if not idx_matches:
            return self._make_result(
                kernel, src, params, "Cannot locate proven index line",
            )

        m = idx_matches[0]
        match_end = m.end()
        after_src = src[match_end:]
        after_stripped = after_src.lstrip()

        if after_stripped.startswith("{"):
            brace_pos_in_after = after_src.index("{")
            absolute_brace = match_end + brace_pos_in_after
            transformed = (
                src[:match_end]
                + "\n    "
                + coarsened
                + src[absolute_brace:]
            )
        elif after_stripped.startswith("if"):
            if_end = _find_if_block_end(after_src, match_end)
            if if_end >= 0:
                block_text = src[match_end:if_end]
                transformed = (
                    src[:match_end]
                    + " {\n    "
                    + coarsened
                    + "\n"
                    + block_text.strip()
                    + "\n    }\n"
                    + src[if_end:]
                )
            else:
                return self._make_result(
                    kernel, src, params, "Cannot locate if-block boundary"
                )
        else:
            semi_pos = _find_statement_end(after_src, match_end)
            if semi_pos >= 0:
                stmt = src[match_end:semi_pos].strip()
                transformed = (
                    src[:match_end]
                    + "\n    "
                    + coarsened
                    + " {\n    "
                    + stmt
                    + "\n    }\n"
                    + src[semi_pos:]
                )
            else:
                return self._make_result(
                    kernel, src, params, "Cannot locate statement boundary"
                )

        warnings.append(
            f"Thread coarsening factor={factor}: each thread processes up to "
            f"{factor} elements. Grid size should be reduced by {factor}x. "
            f"Proven: no atomics, no sync, no shared memory, no early returns."
        )
        result = self._make_result(
            kernel, transformed, params,
            f"Thread coarsening factor={factor}",
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
        if "gridDim" not in src:
            return False
        if "for" not in src:
            return False
        return True


def _find_if_block_end(source_after: str, base_pos: int) -> int:
    paren_start = source_after.find("(")
    if paren_start < 0:
        return -1
    depth = 0
    paren_end = -1
    for i in range(paren_start, len(source_after)):
        if source_after[i] == "(":
            depth += 1
        elif source_after[i] == ")":
            depth -= 1
            if depth == 0:
                paren_end = i
                break
    if paren_end < 0:
        return -1

    after_paren = source_after[paren_end + 1 :].lstrip()
    if after_paren.startswith("{"):
        brace_pos = source_after.index("{", paren_end)
        d = 0
        for i in range(brace_pos, len(source_after)):
            if source_after[i] == "{":
                d += 1
            elif source_after[i] == "}":
                d -= 1
                if d == 0:
                    return base_pos + i + 1
    else:
        semi = source_after.find(";", paren_end)
        if semi >= 0:
            return base_pos + semi + 1
    return -1


def _find_statement_end(source_after: str, base_pos: int) -> int:
    depth = 0
    for i, ch in enumerate(source_after):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == ";" and depth == 0:
            return base_pos + i + 1
        elif ch == "{" and depth == 0:
            return -1
    return -1
