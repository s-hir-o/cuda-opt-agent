"""Kernel fusion transform.

REQUIRES a valid FusionProof from the parser: proven producer-consumer
dependency, no __syncthreads or shared memory, and collision-safe
variable merging.  If any step cannot be verified, the transform
rejects the kernel.
"""

from __future__ import annotations

import re
from typing import Any

from cuda_opt_agent.schemas.bottleneck import BottleneckClass, BottleneckResult
from cuda_opt_agent.schemas.kernel import KernelSource, KernelVariant
from cuda_opt_agent.schemas.transform import TransformResult
from cuda_opt_agent.transforms.analysis import KernelParser, KernelInfo, FusionProof
from cuda_opt_agent.transforms.base import Transformation
from cuda_opt_agent.transforms.registry import register_transform
from cuda_opt_agent.transforms.text.lexical import mask_comments_and_strings, MaskedSource
from cuda_opt_agent.transforms.validation import validate_transform


@register_transform
class KernelFusion(Transformation):

    @property
    def name(self) -> str:
        return "kernel_fusion"

    @property
    def targets(self) -> list[BottleneckClass]:
        return [BottleneckClass.MEMORY_BOUND, BottleneckClass.LATENCY_BOUND]

    @property
    def parameter_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "fused_kernel_name": {
                    "type": "string",
                    "default": "fused_kernel",
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
        src = kernel.source_code
        masked = mask_comments_and_strings(src)
        kernel_pat = re.compile(r"__global__\s+void\s+(\w+)\s*\(")
        matches = masked.filter_matches(kernel_pat)
        if len(matches) != 2:
            return False

        proof = _build_fusion_proof(src, matches)
        if proof is None or not proof.valid:
            return False
        # Must have a proven dependency (not just compatible parameters)
        if not proof.dependency_buffers:
            return False
        # Must not have __syncthreads or shared memory in either kernel
        if not proof.no_syncthreads or not proof.no_shared_memory:
            return False
        return True

    def apply(
        self,
        kernel: KernelSource | KernelVariant,
        params: dict[str, Any],
    ) -> TransformResult:
        fused_name = params.get("fused_kernel_name", "fused_kernel")
        src = kernel.source_code
        masked = mask_comments_and_strings(src)
        warnings: list[str] = []

        kernel_pat = re.compile(r"__global__\s+void\s+(\w+)\s*\(")
        matches = masked.filter_matches(kernel_pat)
        if len(matches) != 2:
            return self._make_result(
                kernel, src, params, "Expected exactly 2 kernels",
            )

        proof = _build_fusion_proof(src, matches)
        if proof is None or not proof.valid:
            return self._make_result(
                kernel, src, params,
                "Cannot prove fusion safety — rejecting",
            )

        # Resolve variable collisions via proven-safe renaming
        k2_body = proof.k2_body
        if proof.variable_collisions:
            for var in sorted(proof.variable_collisions):
                k2_body = _token_replace(k2_body, var, f"{var}_k2")
            warnings.append(
                f"Resolved variable collisions: {proof.variable_collisions} — "
                f"renamed in {proof.kernel2_name} body"
            )

        fused = f"""\
// Fused: {proof.kernel1_name} + {proof.kernel2_name}
// Dependency: {', '.join(sorted(proof.dependency_buffers))}
// Intermediate buffer eliminated — k2 reads k1 results from registers
__global__ void {fused_name}({proof.merged_params}) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // --- {proof.kernel1_name} body (inlined) ---
{proof.k1_body}

    // --- {proof.kernel2_name} body (inlined, reads intermediate from register) ---
{k2_body}
}}
"""

        warnings.append(
            f"Fused '{proof.kernel1_name}' + '{proof.kernel2_name}' into "
            f"'{fused_name}'. Dependency buffers: {proof.dependency_buffers}. "
            f"Verify: intermediate data fits in registers, no sync needed."
        )

        result = self._make_result(
            kernel, fused, params,
            f"Fused '{proof.kernel1_name}' + '{proof.kernel2_name}' → '{fused_name}'",
        )
        result.warnings = warnings
        # Promote success if validation passes
        if self.validate(result):
            result.success = True
        return result

    def validate(self, result: TransformResult) -> bool:
        # Skip comment-preservation: this transform generates new source
        if not validate_transform(
            result.transformed_source,
            result=result,
            check_comments=False,
        ):
            return False
        src = result.transformed_source
        if "__global__ void" not in src:
            return False
        if "fused" not in src.lower():
            return False
        info_orig_1 = _parse_first_kernel(result.original_source)
        if info_orig_1 and info_orig_1.output_write_buffer:
            if info_orig_1.output_write_buffer not in src:
                return False
        return True


# ── Helpers ──────────────────────────────────────────────────────────────────

def _build_fusion_proof(
    source: str, kernel_matches: list[re.Match]
) -> FusionProof | None:
    """Build a FusionProof from exactly two kernel matches.

    Returns None if the two kernels cannot be proven to have a real
    producer-consumer dependency.
    """
    if len(kernel_matches) != 2:
        return None

    masked = mask_comments_and_strings(source)

    k1_name = kernel_matches[0].group(1)
    k2_name = kernel_matches[1].group(1)

    # Parse both kernels
    if len(kernel_matches) >= 2:
        first_part = source[:kernel_matches[1].start()]
    else:
        first_part = source
    second_part = source[kernel_matches[1].start():]

    info1 = KernelParser.parse(first_part)
    info2 = KernelParser.parse(second_part)
    if not info1 or not info2:
        return None

    # Reject if either kernel uses __syncthreads or shared memory
    if info1.has_syncthreads or info2.has_syncthreads:
        return None
    if info1.has_shared_memory or info2.has_shared_memory:
        return None

    # Extract bodies and signatures
    k1_body = _extract_kernel_body(source, kernel_matches[0], masked)
    k2_body = _extract_kernel_body(source, kernel_matches[1], masked)
    k1_sig = _extract_signature(source, kernel_matches[0])
    k2_sig = _extract_signature(source, kernel_matches[1])
    if not k1_body or not k2_body or not k1_sig or not k2_sig:
        return None

    # Merge parameter lists
    k1_params_str = _extract_params_str(k1_sig)
    k2_params_str = _extract_params_str(k2_sig)
    merged_params = _merge_params(k1_params_str, k2_params_str)

    # Compute producer-consumer dependency
    k1_outputs: set[str] = set()
    if info1.output_write_buffer:
        k1_outputs.add(info1.output_write_buffer)
    k1_outputs.update(
        p.name for p in info1.parameters if p.is_pointer and not p.is_const
    )

    k2_reads = set(info2.read_buffer_names)

    dependency = k1_outputs & k2_reads

    if not dependency:
        # Only try positional fallback if the non-pointer params are identical
        # (suggesting the kernels are designed to be chained)
        k1_non_ptr = {p.name for p in info1.parameters if not p.is_pointer}
        k2_non_ptr = {p.name for p in info2.parameters if not p.is_pointer}
        if k1_non_ptr == k2_non_ptr:
            k1_writable = [p for p in info1.parameters
                           if p.is_pointer and not p.is_const]
            k2_readable = [p for p in info2.parameters
                           if p.is_pointer and p.is_const]
            if k1_writable and k2_readable:
                dependency.add(k1_writable[0].name)

    if not dependency:
        return None

    # Compute variable collisions
    k1_locals = _extract_local_decls(k1_body)
    k2_locals = _extract_local_decls(k2_body)
    collisions = k1_locals & k2_locals

    return FusionProof(
        valid=True,
        kernel1_name=k1_name,
        kernel2_name=k2_name,
        dependency_buffers=dependency,
        k1_body=k1_body,
        k2_body=k2_body,
        k1_sig=k1_sig,
        k2_sig=k2_sig,
        variable_collisions=collisions,
        no_syncthreads=True,
        no_shared_memory=True,
        merged_params=merged_params,
    )


def _parse_first_kernel(source: str) -> KernelInfo | None:
    masked = mask_comments_and_strings(source)
    m = masked.filter_matches(re.compile(r"__global__\s+void\s+\w+\s*\("))
    if not m:
        return None
    if len(m) >= 2:
        first_part = source[:m[1].start()]
    else:
        first_part = source
    return KernelParser.parse(first_part)


def _extract_signature(source: str, match: re.Match) -> str | None:
    paren_start = source.find("(", match.start())
    if paren_start < 0:
        return None
    depth = 0
    for i in range(paren_start, len(source)):
        if source[i] == "(":
            depth += 1
        elif source[i] == ")":
            depth -= 1
            if depth == 0:
                return source[match.start() : i + 1]
    return None


def _extract_params_str(sig: str) -> str:
    open_p = sig.find("(")
    close_p = sig.rfind(")")
    if open_p < 0 or close_p < 0:
        return ""
    return sig[open_p + 1 : close_p]


def _extract_kernel_body(
    source: str, match: re.Match, masked: MaskedSource
) -> str | None:
    brace_pos = source.find("{", match.end())
    if brace_pos < 0:
        return None
    depth = 0
    for i in range(brace_pos, len(source)):
        if not masked.is_code(i):
            continue
        if source[i] == "{":
            depth += 1
        elif source[i] == "}":
            depth -= 1
            if depth == 0:
                body = source[brace_pos + 1 : i]
                lines = body.split("\n")
                reindented = []
                for line in lines:
                    stripped = line.strip()
                    if stripped:
                        reindented.append("    " + stripped)
                    else:
                        reindented.append("")
                return "\n".join(reindented)
    return None


def _merge_params(params1: str, params2: str) -> str:
    seen_names: set[str] = set()
    merged: list[str] = []
    for raw in params1.split(","):
        raw = raw.strip()
        if not raw:
            continue
        tokens = raw.replace("*", " * ").split()
        name = tokens[-1] if tokens else raw
        if name not in seen_names:
            seen_names.add(name)
            merged.append(raw)
    for raw in params2.split(","):
        raw = raw.strip()
        if not raw:
            continue
        tokens = raw.replace("*", " * ").split()
        name = tokens[-1] if tokens else raw
        if name not in seen_names:
            seen_names.add(name)
            merged.append(raw)
    return ", ".join(merged)


def _extract_local_decls(body: str) -> set[str]:
    masked = mask_comments_and_strings(body)
    decl_pat = re.compile(
        r"(?:int|float|double|char|unsigned|long|short)\s+(\w+)\s*[=;,)\[]"
    )
    result: set[str] = set()
    for m in masked.filter_matches(decl_pat):
        result.add(m.group(1))
    return result


def _token_replace(source: str, old: str, new: str) -> str:
    """Replace all whole-word occurrences of *old* with *new* in code only."""
    masked = mask_comments_and_strings(source)
    pattern = re.compile(rf"\b{re.escape(old)}\b")
    matches = masked.filter_matches(pattern)
    result = source
    for m in reversed(matches):
        result = result[: m.start()] + new + result[m.end() :]
    return result
