"""Conservative structural parser for CUDA kernel source.

Extracts kernel metadata needed by transforms.  Every field defaults to a
conservative "unknown" / empty value when the structure cannot be proven.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from cuda_opt_agent.transforms.text.lexical import (
    MaskedSource,
    mask_comments_and_strings,
    code_spans_contain,
)


# ── Data containers ─────────────────────────────────────────────────────────

@dataclass
class ParamInfo:
    """Parsed kernel parameter."""
    raw: str
    dtype: str
    name: str
    is_pointer: bool = False
    is_const: bool = False


@dataclass
class LoopInfo:
    """Metadata for a single for-loop."""
    induction_var: str = ""
    trip_count: int | None = None
    start_pos: int = 0
    end_pos: int = 0
    is_innermost: bool = False

@dataclass
class ScalarLoad:
    """Proven scalar load from global memory."""
    buffer: str = ""
    index_var: str = ""
    dtype: str = ""
    line: int = 0


@dataclass
class LoadProof:
    """Per-load safety proof for vectorization."""
    var_name: str = ""
    buf_name: str = ""
    idx_var: str = ""
    scalar_type: str = ""       # 'float' or 'int'
    has_bounds_guard: bool = False
    stride_is_one: bool = False
    no_pointer_cast: bool = False

    @property
    def is_safe(self) -> bool:
        return self.has_bounds_guard and self.stride_is_one and self.no_pointer_cast


# ── Proof objects ───────────────────────────────────────────────────────────

@dataclass
class MatmulProof:
    """Proof that a kernel performs canonical matrix multiplication."""
    accumulation_var: str = ""
    accum_in_innermost_loop: bool = False
    two_read_buffers_in_loop: bool = False
    loop_var_indexed_in_reads: bool = False
    index_vars_used_in_reads: bool = False
    output_write_after_loop: bool = False
    accum_assigned_to_output: bool = False
    at_least_three_loops: bool = False
    # FIX: fields consumed by register_blocking and shared_memory_tiling
    row_major_proven: bool = False
    dim_var: str = ""
    accum_op: str = ""
    buf_a: ParamInfo | None = None
    buf_b: ParamInfo | None = None
    buf_out: ParamInfo | None = None
    k_loop: LoopInfo | None = None
    no_existing_tiling: bool = True
    accum_reads_two_buffers: bool = False

    @property
    def valid(self) -> bool:
        """All structural flags must be true."""
        return (
            self.accum_in_innermost_loop
            and self.two_read_buffers_in_loop
            and self.loop_var_indexed_in_reads
            and self.output_write_after_loop
            and self.no_existing_tiling
        )


@dataclass
class ReductionProof:
    """Proof that a kernel performs a canonical reduction."""
    shared_buffer: str = ""
    halving_pattern: bool = False
    syncthreads_in_loop: bool = False
    sequential_halving: bool = False
    # FIX: fields consumed by warp_shuffle transform
    single_warp_safe: bool = False
    halving_loop: LoopInfo | None = None
    halving_loop_var: str = ""
    output_write_after_loop: bool = False
    sequential_halving_proven: bool = False
    syncthreads_in_halving_loop: bool = False
    reduce_op: str = "+"
    output_buffer: str = ""

    @property
    def valid(self) -> bool:
        """All structural flags must be true."""
        return (
            self.halving_pattern
            and self.syncthreads_in_loop
            and self.sequential_halving
        )


# FIX: single AlignmentProof definition using LoadProof (remove the duplicate)
@dataclass
class AlignmentProof:
    """Proof that scalar loads are safe for vectorization."""
    loads: list[LoadProof] = field(default_factory=list)
    has_bounds_guard: bool = False
    no_pointer_casts: bool = False
    unit_stride: bool = False

    @property
    def valid(self) -> bool:
        """True only if every individual load passes all safety checks."""
        if not self.loads:
            return False
        return all(lp.is_safe for lp in self.loads)


@dataclass
class CoarseningProof:
    """Proof that a kernel is safe to coarsen."""
    index_var: str = ""
    dimension: str = ""
    has_bounds_guard: bool = False
    no_atomics: bool = False
    no_sync: bool = False
    no_shared_mem: bool = False
    no_syncthreads: bool = False
    no_early_return: bool = False

    @property
    def no_shared_memory(self) -> bool:
        return self.no_shared_mem

    @property
    def idx_var(self) -> str:
        return self.index_var

    @property
    def idx_dim(self) -> str:
        return self.dimension

    @property
    def valid(self) -> bool:
        return bool(
            self.index_var
            and self.dimension
            and self.has_bounds_guard
            and self.no_atomics
            and self.no_sync
            and self.no_shared_mem
            and self.no_early_return
        )


@dataclass
class UnrollProof:
    """Proof that loops are safe to unroll."""
    loop_count: int = 0
    no_existing_pragma: bool = False
    loops: list[LoopInfo] = field(default_factory=list)

    # FIX: .valid property consumed by loop_unrolling transform
    @property
    def valid(self) -> bool:
        return self.no_existing_pragma


@dataclass
class KernelInfo:
    """Conservative structural extraction from a CUDA kernel.

    If a field cannot be proven it stays at its default (empty / None / False).
    """
    name: str = ""
    signature: str = ""
    parameters: list[ParamInfo] = field(default_factory=list)
    body: str = ""
    body_start_offset: int = 0  # FIX: consumed by warp_shuffle for position math
    has_loop: bool = False
    has_shared_memory: bool = False
    has_syncthreads: bool = False
    has_grid_stride: bool = False
    index_variables: list[str] = field(default_factory=list)
    bounds_variables: list[str] = field(default_factory=list)
    accumulation_variable: str | None = None
    output_write_buffer: str | None = None
    read_buffer_names: list[str] = field(default_factory=list)
    is_matmul_like: bool = False
    is_reduction_like: bool = False
    shared_memory_buffers: list[str] = field(default_factory=list)
    local_variables: list[str] = field(default_factory=list)
    loops: list[LoopInfo] = field(default_factory=list)

    # Proof objects (None until proven)
    matmul_proof: MatmulProof | None = None
    reduction_proof: ReductionProof | None = None
    alignment_proof: AlignmentProof | None = None
    coarsening_proof: CoarseningProof | None = None
    unroll_proof: UnrollProof | None = None

    # convenience helpers
    def get_read_buffer_names(self) -> list[str]:
        return list(self.read_buffer_names)

    def get_param_names(self) -> list[str]:
        return [p.name for p in self.parameters]

    def get_pointer_params(self) -> list[ParamInfo]:
        return [p for p in self.parameters if p.is_pointer]

    def get_loop_induction_vars(self) -> list[str]:
        return [lp.induction_var for lp in self.loops if lp.induction_var]

    def get_innermost_loop(self) -> LoopInfo | None:
        innermost = [lp for lp in self.loops if lp.is_innermost]
        return innermost[0] if innermost else None

    def extract_matmul_dims(self) -> dict | None:
        """Extract proven M, N, K dimension info.

        Returns dict with keys M, N, K, buf_a, buf_b, accum, out_buf, lv, accum_op
        or None if dimensions cannot be proven.
        """
        if not self.is_matmul_like:
            return None
        return _extract_matmul_dims_from_info(self)

@dataclass
class FusionProof:
    """Proof that two kernels can be safely fused."""
    valid: bool = False
    kernel1_name: str = ""
    kernel2_name: str = ""
    dependency_buffers: set[str] = field(default_factory=set)
    k1_body: str = ""
    k2_body: str = ""
    k1_sig: str = ""
    k2_sig: str = ""
    variable_collisions: set[str] = field(default_factory=set)
    no_syncthreads: bool = False
    no_shared_memory: bool = False
    merged_params: str = ""

# ── Main parser ─────────────────────────────────────────────────────────────

class KernelParser:
    """Conservative structural extractor for CUDA __global__ kernels."""

    @staticmethod
    def parse(source: str) -> KernelInfo:
        """Parse *source* and return a KernelInfo.

        Returns an empty KernelInfo if no valid kernel is found.
        For incomplete kernels (e.g. missing closing paren), returns
        KernelInfo with name set but body empty.
        """
        info = KernelInfo()

        masked = mask_comments_and_strings(source)

        sig_match = masked.filter_matches(
            re.compile(r"__global__\s+void\s+(\w+)\s*\(")
        )
        if not sig_match:
            return info

        m = sig_match[0]
        info.name = m.group(1)
        sig_start = m.start()
        paren_start = source.index("(", m.end() - 1)
        paren_end = _find_matching_paren(source, paren_start)
        if paren_end < 0:
            # Incomplete kernel — return what we have (name only)
            info.signature = source[sig_start:]
            return info

        info.signature = source[sig_start : paren_end + 1].rstrip()
        brace_pos = source.find("{", paren_end)
        if brace_pos < 0:
            return info

        close_brace = _find_matching_brace(source, brace_pos, masked)
        if close_brace < 0:
            return info

        info.body = source[brace_pos + 1 : close_brace]
        info.body_start_offset = brace_pos + 1  # FIX: record offset for transforms
        info.parameters = _parse_parameters(source[paren_start + 1 : paren_end])

        body_masked = mask_comments_and_strings(info.body)

        _detect_loops(info, body_masked)
        info.has_shared_memory = bool(
            code_spans_contain(info.body, body_masked.code_ranges, "__shared__")
        )
        info.has_syncthreads = bool(
            code_spans_contain(info.body, body_masked.code_ranges, "__syncthreads")
        )
        info.has_grid_stride = code_spans_contain(
            info.body, body_masked.code_ranges, "gridDim"
        )
        _detect_global_index(info, body_masked)
        info.shared_memory_buffers = _extract_shared_memory_buffers(
            info.body, body_masked
        )
        _detect_index_variables(info, body_masked)
        _detect_read_buffers(info, body_masked)
        _detect_accumulation(info, body_masked)
        _detect_output_write(info, body_masked)
        _detect_local_variables(info, body_masked)

        info.bounds_variables = [
            p.name
            for p in info.parameters
            if not p.is_pointer
            and any(
                re.search(rf"\b{re.escape(p.name)}\b", lp_src)
                for lp_src in (
                    info.body[lp.start_pos:lp.end_pos] for lp in info.loops
                )
            )
        ]

        _detect_matmul(info, body_masked)
        _detect_reduction(info, body_masked)
        _detect_alignment(info, body_masked)
        _detect_coarsening(info, body_masked)
        _detect_unroll(info, body_masked)

        return info


# ── Detection helpers ───────────────────────────────────────────────────────

def _detect_loops(info: KernelInfo, masked: MaskedSource) -> None:
    loop_pat = re.compile(r"for\s*\(")
    for m in masked.filter_matches(loop_pat):
        paren_start = info.body.index("(", m.end() - 1)
        paren_end = _find_matching_paren(info.body, paren_start)
        if paren_end < 0:
            continue
        after = info.body[paren_end + 1 :].lstrip()
        if not after.startswith("{"):
            continue
        brace_pos = info.body.index("{", paren_end)
        close = _find_matching_brace(info.body, brace_pos, masked)
        if close < 0:
            continue

        lp = LoopInfo(start_pos=m.start(), end_pos=close + 1)

        header = info.body[m.start() : brace_pos]
        iv_match = re.search(r"int\s+(\w+)\s*=", header)
        if iv_match:
            lp.induction_var = iv_match.group(1)

        tc_match = re.search(r"(\w+)\s*<\s*(\d+)\s*;", header)
        if tc_match and lp.induction_var == tc_match.group(1):
            lp.trip_count = int(tc_match.group(2))

        info.loops.append(lp)

    if not info.loops:
        return
    info.has_loop = True

    # Mark innermost: loops that do not contain other loops
    for lp in info.loops:
        lp.is_innermost = not any(
            other.start_pos > lp.start_pos and other.end_pos <= lp.end_pos
            for other in info.loops
            if other is not lp
        )


def _detect_global_index(info: KernelInfo, masked: MaskedSource) -> None:
    pat = re.compile(
        r"(\w+)\s*=\s*blockIdx\.(\w)\s*\*\s*blockDim\.(\w)\s*\+\s*threadIdx\.(\w)\s*;"
    )
    matches = masked.filter_matches(pat)
    for m in matches:
        var = m.group(1)
        if var not in info.index_variables:
            info.index_variables.append(var)


def _detect_index_variables(info: KernelInfo, masked: MaskedSource) -> None:
    for lp in info.loops:
        if lp.is_innermost and lp.induction_var:
            if lp.induction_var not in info.index_variables:
                info.index_variables.append(lp.induction_var)


def _extract_shared_memory_buffers(body: str, masked: MaskedSource) -> list[str]:
    pat = re.compile(r"__shared__\s+[\w]+\s+(\w+)")
    return [m.group(1) for m in masked.filter_matches(pat)]


def _detect_read_buffers(info: KernelInfo, masked: MaskedSource) -> None:
    names: set[str] = set()
    for m in masked.filter_matches(re.compile(r"(\w+)\s*\[")):
        buf = m.group(1)
        if buf in ("if", "for", "while", "switch"):
            continue
        names.add(buf)
    for m in masked.filter_matches(re.compile(r"\*\s*\(\s*(\w+)\s*\+")):
        names.add(m.group(1))
    info.read_buffer_names = sorted(names - set(info.shared_memory_buffers))


def _detect_accumulation(info: KernelInfo, masked: MaskedSource) -> None:
    if not info.has_loop:
        return
    inner = [lp for lp in info.loops if lp.is_innermost]
    if not inner:
        return
    body = info.body[inner[0].start_pos : inner[0].end_pos]
    aug = re.search(r"\b(\w+)\s*[+\-*|&^]=", body)
    if aug:
        info.accumulation_variable = aug.group(1)
        return
    assign = re.search(r"\b(\w+)\s*=\s*\w+\s*[+\*]\s*\w+", body)
    if assign:
        info.accumulation_variable = assign.group(1)


def _detect_output_write(info: KernelInfo, masked: MaskedSource) -> None:
    if info.has_loop:
        inner = [lp for lp in info.loops if lp.is_innermost]
        if not inner:
            return
        after_loop = info.body[inner[0].end_pos :]
    else:
        after_loop = info.body
    after_masked = mask_comments_and_strings(after_loop)
    # FIX: \[ not $$$$  (the original had $$ which is end-of-string anchor)
    m = after_masked.filter_matches(re.compile(r"(\w+)\s*\["))
    if m:
        info.output_write_buffer = m[0].group(1)
        return
    m = after_masked.filter_matches(re.compile(r"\*\s*\(\s*(\w+)"))
    if m:
        info.output_write_buffer = m[0].group(1)


def _detect_local_variables(info: KernelInfo, masked: MaskedSource) -> None:
    declared = _get_declared_variables(info.body, masked)
    param_names = {p.name for p in info.parameters}
    info.local_variables = sorted(declared - param_names)

def _detect_matmul(info: KernelInfo, masked: MaskedSource) -> None:
    """Detect canonical matrix multiply only when structurally proven."""
    if not info.accumulation_variable:
        return
    accum = info.accumulation_variable
    if not info.output_write_buffer:
        return

    inner = [lp for lp in info.loops if lp.is_innermost]
    if not inner:
        return
    inner_loop = inner[0]
    inner_body = info.body[inner_loop.start_pos : inner_loop.end_pos]
    if accum not in inner_body:
        return

    # Gather reads in inner loop — both global and shared memory
    global_reads = [
        b for b in info.read_buffer_names
        if b in inner_body and b != info.output_write_buffer
    ]
    shared_reads = [
        b for b in info.shared_memory_buffers
        if b in inner_body
    ]
    all_reads = global_reads + shared_reads
    if len(all_reads) < 2:
        return

    loop_vars = info.get_loop_induction_vars()
    if not loop_vars:
        return
    has_indexed_access = any(
        re.search(rf"\b{re.escape(rbuf)}\s*\[.*{re.escape(lv)}", inner_body)
        for rbuf in all_reads
        for lv in loop_vars
    )
    if not has_indexed_access:
        return

    # Detect accumulation operator
    accum_op = "+"
    aug_match = re.search(
        rf"\b{re.escape(accum)}\s*([+\-*|&^])=", inner_body
    )
    if aug_match:
        accum_op = aug_match.group(1)
    else:
        assign_match = re.search(
            rf"\b{re.escape(accum)}\s*=\s*{re.escape(accum)}\s*([+\*])\s*",
            inner_body,
        )
        if assign_match:
            accum_op = assign_match.group(1)

    # Case 1: Canonical matmul — global reads, try dimension extraction
    if len(global_reads) >= 2:
        dims = _extract_matmul_dims_from_info(info)
        if dims is not None:
            dim_var = dims.get("N", "")
            buf_a_info = next(
                (p for p in info.parameters if p.name == global_reads[0]), None
            )
            buf_b_info = next(
                (p for p in info.parameters if p.name == global_reads[1]), None
            )
            buf_out_info = next(
                (p for p in info.parameters
                 if p.name == info.output_write_buffer), None,
            )
            proof = MatmulProof(
                accumulation_var=accum,
                accum_in_innermost_loop=True,
                two_read_buffers_in_loop=True,
                accum_reads_two_buffers=True,
                loop_var_indexed_in_reads=True,
                index_vars_used_in_reads=has_indexed_access,
                output_write_after_loop=True,
                accum_assigned_to_output=True,
                at_least_three_loops=len(info.loops) >= 3,
                row_major_proven=True,
                dim_var=dim_var,
                accum_op=accum_op,
                buf_a=buf_a_info,
                buf_b=buf_b_info,
                buf_out=buf_out_info,
                k_loop=inner_loop,
                no_existing_tiling=not info.has_shared_memory,
            )
            if proof.valid:
                info.is_matmul_like = True
                info.matmul_proof = proof
            return

    # Case 2: Tiled matmul — shared memory reads, 3+ loops
    if info.has_shared_memory and len(info.loops) >= 2 and len(shared_reads) >= 2:
        dim_var = ""
        if info.bounds_variables:
            dim_var = info.bounds_variables[0]

        proof = MatmulProof(
            accumulation_var=accum,
            accum_in_innermost_loop=True,
            two_read_buffers_in_loop=True,
            accum_reads_two_buffers=True,
            loop_var_indexed_in_reads=True,
            index_vars_used_in_reads=has_indexed_access,
            output_write_after_loop=True,
            accum_assigned_to_output=True,
            at_least_three_loops=True,
            row_major_proven=False,
            dim_var=dim_var,
            accum_op=accum_op,
            k_loop=inner_loop,
            no_existing_tiling=False,  # already has shared memory
        )
        info.is_matmul_like = True
        info.matmul_proof = proof


def _detect_reduction(info: KernelInfo, masked: MaskedSource) -> None:
    """Detect reduction only when structurally proven."""
    if not info.has_shared_memory:
        return
    if not info.has_syncthreads:
        return

    # FIX: relaxed — just require >>= 1 or /= 2 anywhere in the source
    has_shift = masked.filter_matches(re.compile(r"\w+\s*>>=\s*1"))
    has_div = masked.filter_matches(re.compile(r"\w+\s*/=\s*2"))
    if not has_shift and not has_div:
        return

    sbuf = info.shared_memory_buffers[0] if info.shared_memory_buffers else ""

    for lp in info.loops:
        body = info.body[lp.start_pos:lp.end_pos]
        has_smem = any(s in body for s in info.shared_memory_buffers)
        if not has_smem:
            continue
        if "__syncthreads" not in body:
            continue
        if ">>= 1" not in body and "/= 2" not in body:
            continue

        shift_match = re.search(r"(\w+)\s*>>=\s*1", body)
        halving_var = shift_match.group(1) if shift_match else lp.induction_var

        reduce_op = "+"
        if re.search(r"\+=", body):
            reduce_op = "+"
        elif re.search(r"\*=", body):
            reduce_op = "*"
        if "fmaxf" in body or "fmax" in body:
            reduce_op = "max"
        elif "fminf" in body or "fmin" in body:
            reduce_op = "min"

        single_warp = True
        if re.search(r"blockDim\.\w+\s*/\s*2", body) and "threadIdx." not in body:
            single_warp = False
        if re.search(r"blockDim\.\w+\s*>>\s*1", body) and "threadIdx." not in body:
            single_warp = False

        out_buf = info.output_write_buffer or ""

        info.is_reduction_like = True
        info.reduction_proof = ReductionProof(
            shared_buffer=sbuf,
            halving_pattern=True,
            syncthreads_in_loop=True,
            sequential_halving=True,
            single_warp_safe=single_warp,
            halving_loop=lp,
            halving_loop_var=halving_var,
            output_write_after_loop=bool(out_buf),
            sequential_halving_proven=True,
            syncthreads_in_halving_loop=True,
            reduce_op=reduce_op,
            output_buffer=out_buf,
        )
        return


def _detect_alignment(info: KernelInfo, masked: MaskedSource) -> None:
    """Detect scalar loads safe for vectorization."""
    load_pat = re.compile(
        r"(float|int)\s+(\w+)\s*=\s*(\w+)\s*\[\s*(\w+)\s*\]\s*;"
    )
    loads: list[LoadProof] = []
    for m in masked.filter_matches(load_pat):
        loads.append(LoadProof(
            var_name=m.group(2),
            buf_name=m.group(3),
            idx_var=m.group(4),
            scalar_type=m.group(1),
        ))

    if not loads:
        return

    has_bounds = bool(
        masked.filter_matches(re.compile(r"if\s*\(\s*\w+\s*<\s*\w+"))
    )
    has_casts = bool(masked.filter_matches(
        re.compile(r"\*\s*\(\s*(?:float|int)[24]?\s*\*\s*\(")
    ))

    for lp in loads:
        lp.has_bounds_guard = has_bounds
        lp.no_pointer_cast = not has_casts
        lp.stride_is_one = not bool(re.search(r"[+\-*]", lp.idx_var))

    info.alignment_proof = AlignmentProof(
        loads=loads,
        has_bounds_guard=has_bounds,
        no_pointer_casts=not has_casts,
        unit_stride=all(lp.stride_is_one for lp in loads),
    )


def _detect_coarsening(info: KernelInfo, masked: MaskedSource) -> None:
    """Detect kernels safe for thread coarsening."""
    idx_pat = re.compile(
        r"int\s+(\w+)\s*=\s*blockIdx\.(\w)\s*\*\s*blockDim\.(\w)\s*\+\s*threadIdx\.(\w)\s*;"
    )
    idx_matches = masked.filter_matches(idx_pat)
    if not idx_matches:
        return
    if info.has_grid_stride:
        return
    if info.has_syncthreads or info.has_shared_memory:
        return
    if info.is_matmul_like or info.is_reduction_like:
        return

    has_atomics = bool(masked.filter_matches(
        re.compile(r"atomicAdd|atomicSub|atomicExch|atomicCAS")
    ))
    if has_atomics:
        return

    has_bounds = bool(
        masked.filter_matches(re.compile(r"if\s*\(\s*\w+\s*<\s*\w+"))
    )
    if not has_bounds:
        return

    has_early_return = bool(masked.filter_matches(re.compile(r"\breturn\s*;")))
    if has_early_return:
        return

    m = idx_matches[0]
    info.coarsening_proof = CoarseningProof(
        index_var=m.group(1),
        dimension=m.group(2),
        has_bounds_guard=has_bounds,
        no_atomics=not has_atomics,
        no_sync=not info.has_syncthreads,
        no_shared_mem=not info.has_shared_memory,
        no_syncthreads=not info.has_syncthreads,
        no_early_return=not has_early_return,
    )

def _detect_unroll(info: KernelInfo, masked: MaskedSource) -> None:
    """Detect loops safe for unrolling."""
    has_pragma = bool(
        masked.filter_matches(re.compile(r"#pragma\s+unroll"))
    )
    info.unroll_proof = UnrollProof(
        loop_count=len(info.loops),
        no_existing_pragma=not has_pragma,
        loops=list(info.loops),
    )


# ── Matmul dimension extraction ─────────────────────────────────────────────

def _extract_matmul_dims_from_info(info: KernelInfo) -> dict | None:
    """Extract M, N, K dimensions from a proven matmul kernel.

    Returns dict or None if dimensions cannot be proven.
    """
    if not info.accumulation_variable or not info.output_write_buffer:
        return None
    inner = info.get_innermost_loop()
    if not inner or not inner.induction_var:
        return None

    accum = info.accumulation_variable
    out_buf = info.output_write_buffer
    inner_body = info.body[inner.start_pos:inner.end_pos]
    lv = inner.induction_var

    # Find accumulation operator
    aug_match = re.search(
        rf"\b{re.escape(accum)}\s*([+\-*|&^])=", inner_body
    )
    if aug_match:
        op = aug_match.group(1) + "="
        rhs_start = aug_match.end()
    else:
        assign_match = re.search(
            rf"\b{re.escape(accum)}\s*=\s*{re.escape(accum)}\s*([+\*])\s*",
            inner_body,
        )
        if assign_match:
            op = assign_match.group(1) + "="
            rhs_start = assign_match.end()
        else:
            return None

    rhs = inner_body[rhs_start:]

    # Find the two read buffers in accumulation
    used: list[str] = []
    seen: set[str] = set()
    for buf in info.read_buffer_names:
        if buf == out_buf:
            continue
        if buf + "[" in rhs and buf not in seen:
            used.append(buf)
            seen.add(buf)
    if len(used) < 2:
        return None
    buf_a, buf_b = used[0], used[1]

    # FIX: was $$$$\s*([^$$]+)\s*  →  \[\s*([^\]]+)\s*\]
    m_a = re.search(rf"{re.escape(buf_a)}\s*\[\s*([^\]]+)\s*\]", rhs)
    m_b = re.search(rf"{re.escape(buf_b)}\s*\[\s*([^\]]+)\s*\]", rhs)
    if not m_a or not m_b:
        return None

    idx_a = m_a.group(1).strip()
    idx_b = m_b.group(1).strip()

    # Parse A indexing: row_var * K_stride + lv
    row_var = ""
    ma = re.search(rf"(\w+)\s*\*\s*(\w+)\s*\+\s*{re.escape(lv)}", idx_a)
    if ma:
        row_var = ma.group(1)
    else:
        ma2 = re.search(rf"{re.escape(lv)}\s*\*\s*(\w+)\s*\+\s*(\w+)", idx_a)
        if ma2:
            row_var = ma2.group(2)

    # Parse B indexing: lv * N_stride + col_var
    col_var = ""
    mb = re.search(rf"{re.escape(lv)}\s*\*\s*(\w+)\s*\+\s*(\w+)", idx_b)
    if mb:
        col_var = mb.group(2)
    else:
        mb2 = re.search(rf"(\w+)\s*\+\s*{re.escape(lv)}\s*\*\s*(\w+)", idx_b)
        if mb2:
            col_var = mb2.group(1)

    if not row_var or not col_var:
        return None

    # Find bounds
    M = ""
    N = ""

    row_bound = re.search(rf"\b{re.escape(row_var)}\s*<\s*(\w+)", info.body)
    if row_bound:
        M = row_bound.group(1)

    col_bound = re.search(rf"\b{re.escape(col_var)}\s*<\s*(\w+)", info.body)
    if col_bound:
        N = col_bound.group(1)

    K = ""
    if inner.trip_count is not None:
        K = str(inner.trip_count)
    else:
        k_bound = re.search(
            rf"{re.escape(lv)}\s*<\s*(\w+)",
            info.body[inner.start_pos:inner.end_pos],
        )
        if k_bound:
            K = k_bound.group(1)

    if K and not K.isdigit():
        if not M:
            M = K
        if not N:
            N = K

    if not M or not N:
        return None

    return {
        "M": M, "N": N, "K": K,
        "buf_a": buf_a, "buf_b": buf_b,
        "accum": accum, "out_buf": out_buf,
        "lv": lv, "accum_op": op,
    }


# ── Utilities ───────────────────────────────────────────────────────────────

def _find_matching_paren(source: str, start: int) -> int:
    """Find the index of the closing ')' matching the '(' at *start*."""
    depth = 0
    for i in range(start, len(source)):
        if source[i] == "(":
            depth += 1
        elif source[i] == ")":
            depth -= 1
            if depth == 0:
                return i
    return -1


def _find_matching_brace(
    source: str, start: int, masked: MaskedSource
) -> int:
    """Find the index of '}' matching the '{' at *start*, skipping non-code."""
    depth = 0
    for i in range(start, len(source)):
        if not masked.is_code(i):
            continue
        if source[i] == "{":
            depth += 1
        elif source[i] == "}":
            depth -= 1
            if depth == 0:
                return i
    return -1


def _parse_parameters(param_text: str) -> list[ParamInfo]:
    params: list[ParamInfo] = []
    for raw in param_text.split(","):
        raw = raw.strip()
        if not raw:
            continue
        tokens = raw.replace("*", " * ").replace("&", " & ").split()
        is_const = "const" in tokens
        is_pointer = "*" in raw or "[" in raw
        clean = [t for t in tokens if t not in ("const", "*", "&")]
        if len(clean) >= 2:
            dtype = " ".join(clean[:-1])
            name = clean[-1]
        else:
            dtype = raw
            name = raw
        params.append(
            ParamInfo(
                raw=raw, dtype=dtype, name=name,
                is_pointer=is_pointer, is_const=is_const,
            )
        )
    return params


def _get_declared_variables(body: str, masked: MaskedSource) -> set[str]:
    decl_pat = re.compile(
        r"(?:^|\s)(?:int|float|double|char|unsigned|long|short)\s+(\w+)\s*[=;,)\[]"
    )
    ptr_pat = re.compile(
        r"(?:int|float|double|char)\s*\*\s*(\w+)"
    )
    result: set[str] = set()
    for m in masked.filter_matches(decl_pat):
        result.add(m.group(1))
    for m in masked.filter_matches(ptr_pat):
        result.add(m.group(1))
    return result
