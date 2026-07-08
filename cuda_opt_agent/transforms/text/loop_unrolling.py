"""Loop unrolling transform.

REQUIRES a valid UnrollProof from the parser.  Adds ``#pragma unroll``
before detected for-loops in code regions only.  Rejects kernels that
already contain unroll pragmas.
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
class LoopUnrolling(Transformation):

    @property
    def name(self) -> str:
        return "loop_unrolling"

    @property
    def targets(self) -> list[BottleneckClass]:
        return [BottleneckClass.COMPUTE_BOUND, BottleneckClass.LATENCY_BOUND]

    @property
    def parameter_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "factor": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 32,
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
        proof = info.unroll_proof
        # Require a valid UnrollProof
        if proof is None or not proof.valid:
            return False
        if not proof.no_existing_pragma:
            return False
        if proof.loop_count == 0:
            return False
        return True

    def apply(
        self,
        kernel: KernelSource | KernelVariant,
        params: dict[str, Any],
    ) -> TransformResult:
        factor = params.get("factor", 4)
        src = kernel.source_code
        warnings: list[str] = []

        masked = mask_comments_and_strings(src)

        for_pat = re.compile(r"^(\s*)(for\s*\()", re.MULTILINE)
        matches = masked.filter_matches(for_pat)

        if not matches:
            return self._make_result(
                kernel, src, params, "No for-loops found in code",
            )

        lines = src.split("\n")
        line_starts = _build_line_starts(src)

        pragma_insertions: list[tuple[int, str, int]] = []

        for m in matches:
            pos = m.start()
            line_idx = _pos_to_line(pos, line_starts)
            indent = m.group(1)

            if line_idx > 0:
                prev_line = lines[line_idx - 1].strip()
                if prev_line.startswith("#pragma unroll"):
                    continue

            loop_line = lines[line_idx]
            tc_match = re.search(r"(\w+)\s*<\s*(\d+)", loop_line)
            pragma_factor = factor
            if tc_match:
                trip_count = int(tc_match.group(2))
                if trip_count <= 32:   # FIX: was `<= factor`
                    pragma_factor = trip_count

            pragma_insertions.append((line_idx, indent, pragma_factor))

        if not pragma_insertions:
            return self._make_result(
                kernel, src, params, "All loops already unrolled",
            )

        for line_idx, indent, pf in reversed(pragma_insertions):
            lines.insert(line_idx, f"{indent}#pragma unroll {pf}")

        transformed = "\n".join(lines)
        warnings.append(
            f"Added #pragma unroll to {len(pragma_insertions)} loop(s) "
            f"(proven: no existing pragmas)"
        )
        result = self._make_result(
            kernel, transformed, params,
            f"Loop unroll factor={factor}",
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
        masked = mask_comments_and_strings(src)
        pragmas = masked.filter_matches(re.compile(r"#pragma\s+unroll"))
        if not pragmas:
            return False
        lines = src.split("\n")
        for pm in pragmas:
            pos = pm.start()
            line_idx = _pos_to_line(pos, _build_line_starts(src))
            for j in range(line_idx + 1, min(line_idx + 3, len(lines))):
                stripped = lines[j].strip()
                if stripped:
                    if not stripped.startswith("for"):
                        return False
                    break
        return True


def _build_line_starts(source: str) -> list[int]:
    starts = [0]
    for i, ch in enumerate(source):
        if ch == "\n" and i + 1 < len(source):
            starts.append(i + 1)
    return starts


def _pos_to_line(pos: int, line_starts: list[int]) -> int:
    for idx in range(len(line_starts) - 1, -1, -1):
        if pos >= line_starts[idx]:
            return idx
    return 0
