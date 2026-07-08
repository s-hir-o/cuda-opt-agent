# CHANGELOG — Layer 4: Transform Engine

## What was changed

### analysis.py — Structured proof objects

The parser now builds **proof objects** alongside the existing boolean flags.
Each proof is only populated when the corresponding pattern is STRICTLY
verified — never by guesswork.  Transforms require valid proof objects
before generating code.

**New proof types:**
- `MatmulProof` — row-major matmul with verified accumulation, buffer roles,
  loop roles, dimension variable, and row-major indexing
- `ReductionProof` — shared-memory sequential halving with verified operator,
  output write, and single-warp safety
- `AlignmentProof` — per-load safety checks: bounds guard for the specific
  index variable, stride-1 access, no pointer casts
- `CoarseningProof` — single-index pattern with no atomics, no sync, no
  shared memory, no early returns, no grid-stride
- `UnrollProof` — loops exist and no `#pragma unroll` is already present
- `FusionProof` — built by kernel_fusion.py itself from two parsed kernels

**New helper types:**
- `BufferRole` — name + access indices + pointer status
- `LoopProof` — induction var + trip count + nesting level + role
- `LoadProof` — per-load safety flags

### shared_memory_tiling.py — Proof-gated

Now requires a valid `MatmulProof` with all verification flags True.
Uses proven `dim_var`, `accum_op`, buffer names, and loop variable names.
Rejects anything the parser cannot fully verify.

### register_blocking.py — Proof-gated

Same requirement: valid `MatmulProof` with row-major proven and dimension
available.  Uses proven accumulation operator in compute block.

### vectorized_loads.py — Per-load proof

Requires a valid `AlignmentProof` where every individual `LoadProof` passes
all three safety checks (bounds guard for the specific index, stride-1, no
pointer casts).  Name-conflict check prevents temp variable collisions.

### thread_coarsening.py — Proof-gated

Requires a valid `CoarseningProof`.  No heuristics at apply-time.

### warp_shuffle.py — Complete rewrite or fail closed

Requires a valid `ReductionProof` with `single_warp_safe=True`.  The
halving loop initial expression must NOT involve `blockDim` or `gridDim`.
After generating the replacement code, the transform verifies that
`__shfl_down_sync` actually appears in the output and that the output buffer
is preserved.  If any post-rewrite check fails, returns failure.

### kernel_fusion.py — Dependency proof

`_build_fusion_proof()` computes the exact producer-consumer dependency set
(k1 output buffers intersected with k2 read buffers).  If the intersection
is empty, the proof is invalid and fusion is rejected.  Variable collisions
are computed and resolved only when the proof is valid.

### loop_unrolling.py — Proof-gated

Requires a valid `UnrollProof` with `no_existing_pragma=True`.

### base.py — _make_result defaults to success=False

Unvalidated output can no longer be consumed as a completed optimisation.

### validation.py — Masked brace checking

All bracket-type checks now use the masked source so braces inside comments
and strings do not corrupt the count.

### registry.py — Duplicate protection

`register_transform()` rejects duplicate names.  `list_transforms()` and
`transforms_for_bottleneck()` return deterministic (sorted) results.

### lexical.py — Equal-length enforcement

`MaskedSource.transform()` raises `ValueError` if the transform function
returns a string of different length.

## What is intentionally conservative

- **SharedMemoryTiling** and **RegisterBlocking** only support proven
  row-major single-square-dimension matmul.  Transposed, batched, or
  non-square layouts are rejected.
- **VectorizedLoads** requires per-load proof.  Only `.x` component is
  extracted; full bandwidth requires manual restructuring.
- **WarpShuffleReduction** rejects multi-warp reductions.  The halving
  loop initial expression must be a constant, not `blockDim / 2`.
- **ThreadCoarsening** rejects kernels with early returns, atomics,
  syncthreads, shared memory, or grid-stride loops.
- **KernelFusion** requires an exact proven dependency buffer set.
- The parser prefers **false negatives over false positives**.  If any
  verification flag is False, the entire proof is invalid.
- Strict nvcc compilation is opt-in and skipped when nvcc is unavailable.

## Files changed
- `cuda_opt_agent/transforms/analysis.py`
- `cuda_opt_agent/transforms/base.py`
- `cuda_opt_agent/transforms/registry.py`
- `cuda_opt_agent/transforms/validation.py`
- `cuda_opt_agent/transforms/text/lexical.py`
- `cuda_opt_agent/transforms/text/shared_memory_tiling.py`
- `cuda_opt_agent/transforms/text/register_blocking.py`
- `cuda_opt_agent/transforms/text/vectorized_loads.py`
- `cuda_opt_agent/transforms/text/thread_coarsening.py`
- `cuda_opt_agent/transforms/text/warp_shuffle.py`
- `cuda_opt_agent/transforms/text/kernel_fusion.py`
- `cuda_opt_agent/transforms/text/loop_unrolling.py`

## Tests added / expanded
- `test_proof_objects.py` — proof construction for matmul, reduction,
  alignment, coarsening, unroll; verification flags; negative cases
- `test_all_transforms.py` — proof-gated applicability, rejection, round-trip
- `test_registry.py` — registration, lookup, duplicate safety, bottleneck
  filtering, schema validation
- `test_transform_validation.py` — structural checks, placeholder rejection,
  comment/string preservation, result promotion
- `test_lexical.py` — masking, filter_matches, code_spans_contain,
  equal-length enforcement
