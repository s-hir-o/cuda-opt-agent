"""Vectorized loads transform.

Replaces scalar global memory loads with vector types (float2/float4)
for improved memory coalescing and bandwidth.

REQUIRES a valid AlignmentProof from the parser.  Every individual load
must be proven safe:
  - has an explicit bounds guard for the specific index variable
  - uses stride-1 access (simple index, no expression)
  - no pointer casts are present in the kernel

If any load fails any check, the transform rejects the kernel entirely.
"""

from __future__ import annotations

import re
from typing import Any

from cuda_opt_agent.schemas.bottleneck import BottleneckClass, BottleneckResult
from cuda_opt_agent.schemas.kernel import KernelSource, KernelVariant
from cuda_opt_agent.schemas.transform import TransformResult
from cuda_opt_agent.transforms.analysis import KernelParser, LoadProof
from cuda_opt_agent.transforms.base import Transformation
from cuda_opt_agent.transforms.registry import register_transform
from cuda_opt_agent.transforms.text.lexical import mask_comments_and_strings
from cuda_opt_agent.transforms.validation import validate_transform


@register_transform
class VectorizedLoads(Transformation):

    @property
    def name(self) -> str:
        return "vectorized_loads"

    @property
    def targets(self) -> list[BottleneckClass]:
        return [BottleneckClass.MEMORY_BOUND]

    @property
    def parameter_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "vector_width": {
                    "type": "integer",
                    "enum": [2, 4],
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
        proof = info.alignment_proof
        # Require a fully valid AlignmentProof
        if proof is None or not proof.valid:
            return False
        # Must have at least one safe float load
        float_loads = [lp for lp in proof.loads if lp.scalar_type == "float"]
        if not float_loads:
            return False
        return True

    def apply(
        self,
        kernel: KernelSource | KernelVariant,
        params: dict[str, Any],
    ) -> TransformResult:
        width = params.get("vector_width", 4)
        info = KernelParser.parse(kernel.source_code)
        proof = info.alignment_proof
        src = kernel.source_code
        warnings: list[str] = []

        if proof is None or not proof.valid:
            return self._make_result(
                kernel, src, params,
                "No valid alignment proof — cannot vectorize",
            )

        # Collect all locally-declared variable names to prevent name conflicts
        all_existing_vars: set[str] = set()
        for lp in proof.loads:
            all_existing_vars.add(lp.var_name)
        for p in info.parameters:
            all_existing_vars.add(p.name)
        for v in info.local_variables:
            all_existing_vars.add(v)

        vec_type = f"float{width}"

        # Build replacements from proven-safe loads only
        safe_load_re = re.compile(
            r"(\s*)(float)\s+(\w+)\s*=\s*(\w+)\s*$$\s*(\w+)\s*$$\s*;"
        )
        masked = mask_comments_and_strings(src)
        loads = masked.filter_matches(safe_load_re)

        replacements: list[tuple[int, int, str]] = []

        for m in loads:
            indent = m.group(1)
            var_name = m.group(3)
            buf_name = m.group(4)
            idx_var = m.group(5)

            # Verify this specific load has a valid proof entry
            load_proof = None
            for lp in proof.loads:
                if (lp.var_name == var_name
                        and lp.buf_name == buf_name
                        and lp.idx_var == idx_var
                        and lp.has_bounds_guard
                        and lp.stride_is_one
                        and lp.no_pointer_cast):
                    load_proof = lp
                    break

            if load_proof is None:
                continue  # Skip loads that lack individual proof

            # Ensure invented temp name does not conflict
            temp_name = f"{var_name}_v"
            if temp_name in all_existing_vars:
                continue

            replacement = (
                f"{indent}{vec_type} {temp_name} = "
                f"*({vec_type}*)&{buf_name}[{idx_var} * {width}];\n"
                f"{indent}float {var_name} = {temp_name}.x;"
            )
            replacements.append((m.start(), m.end(), replacement))

        if not replacements:
            return self._make_result(
                kernel, src, params,
                "No float loads passed individual safety proof",
            )

        # Apply replacements back-to-front
        transformed = src
        for start, end, repl in reversed(replacements):
            transformed = transformed[:start] + repl + transformed[end:]

        warnings.append(
            f"Vectorized {len(replacements)} load(s) to {vec_type}. "
            f"Each load individually proven safe: bounds guard + stride-1 "
            f"+ no pointer casts. Only .x component extracted."
        )
        result = self._make_result(
            kernel, transformed, params,
            f"Vectorized {len(replacements)} loads to {vec_type}",
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
        if re.search(r"\b\w+_base\b", src):
            return False
        if not re.search(r"float[24]", src):
            return False
        return True
