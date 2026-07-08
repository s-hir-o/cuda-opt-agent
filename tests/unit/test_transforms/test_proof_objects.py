"""Tests that proof objects are correctly built by the structural parser."""

import pytest

from cuda_opt_agent.transforms.analysis import (
    KernelParser,
    MatmulProof,
    ReductionProof,
    AlignmentProof,
    CoarseningProof,
    UnrollProof,
)

MATMUL = """\
__global__ void matmul(float *C, float *A, float *B, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}
"""

MATMUL_WITH_SHARED = """\
__global__ void matmul(float *C, float *A, float *B, int N) {
    __shared__ float As[16][16];
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    for (int k = 0; k < N; k += 16) {
        As[threadIdx.y][threadIdx.x] = A[row * N + k + threadIdx.x];
        __syncthreads();
        for (int j = 0; j < 16; j++) {
            sum += As[threadIdx.y][j] * B[k * j + col];
        }
        __syncthreads();
    }
    C[row * N + col] = sum;
}
"""

AMBIGUOUS_PATTERN = """\
__global__ void weird(float *out, float *a, float *b, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float s = 0.0f;
    for (int i = 0; i < N; i++) {
        s += a[idx * N + i] * b[i];
    }
    out[idx] = s;
}
"""

REDUCTION = """\
__global__ void reduce(float *out, float *in, int N) {
    __shared__ float sdata[32];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = (idx < N) ? in[idx] : 0.0f;
    __syncthreads();
    for (int s = 16; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        out[blockIdx.x] = sdata[0];
    }
}
"""

REDUCTION_MULTI_WARP = """\
__global__ void reduce(float *out, float *in, int N) {
    __shared__ float sdata[256];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = (idx < N) ? in[idx] : 0.0f;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        out[blockIdx.x] = sdata[0];
    }
}
"""

VEC_ADD = """\
__global__ void add(float *out, float *a, float *b, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        out[idx] = a[idx] + b[idx];
    }
}
"""

SCALAR_LOADS = """\
__global__ void process(float *out, float *a, float *b, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float va = a[idx];
        float vb = b[idx];
        out[idx] = va + vb;
    }
}
"""

UNROLLED = """\
__global__ void k(float *out, float *in, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float val = 0.0f;
        #pragma unroll 4
        for (int i = 0; i < 4; i++) {
            val += in[idx * 4 + i];
        }
        out[idx] = val;
    }
}
"""


class TestMatmulProof:

    def test_valid_matmul_has_proof(self):
        info = KernelParser.parse(MATMUL)
        assert info.matmul_proof is not None
        assert info.matmul_proof.valid is True

    def test_matmul_proof_has_all_fields(self):
        info = KernelParser.parse(MATMUL)
        p = info.matmul_proof
        assert p.accumulation_var == "sum"
        assert p.accum_op == "+"
        assert p.buf_a.name == "A"
        assert p.buf_b.name == "B"
        assert p.buf_out.name == "C"
        assert p.dim_var == "N"
        assert p.row_major_proven is True
        assert p.no_existing_tiling is True

    def test_matmul_with_shared_no_tiling_proof(self):
        info = KernelParser.parse(MATMUL_WITH_SHARED)
        p = info.matmul_proof
        # Has shared memory → no_existing_tiling=False → proof invalid
        assert p is None or p.valid is False

    def test_ambiguous_pattern_no_matmul_proof(self):
        """Two loops with two buffers but not three nested loops → no proof."""
        info = KernelParser.parse(AMBIGUOUS_PATTERN)
        # Not is_matmul_like because only 2 loops
        assert info.matmul_proof is None

    def test_non_matmul_no_proof(self):
        info = KernelParser.parse(VEC_ADD)
        assert info.matmul_proof is None

    def test_matmul_verification_flags_all_true(self):
        info = KernelParser.parse(MATMUL)
        p = info.matmul_proof
        assert p.accum_in_innermost_loop is True
        assert p.accum_reads_two_buffers is True
        assert p.index_vars_used_in_reads is True
        assert p.accum_assigned_to_output is True
        assert p.no_existing_tiling is True
        assert p.row_major_proven is True


class TestReductionProof:

    def test_valid_reduction_has_proof(self):
        info = KernelParser.parse(REDUCTION)
        assert info.reduction_proof is not None
        assert info.reduction_proof.valid is True

    def test_reduction_proof_fields(self):
        info = KernelParser.parse(REDUCTION)
        p = info.reduction_proof
        assert p.shared_buffer == "sdata"
        assert p.reduce_op == "+"
        assert p.output_buffer == "out"
        assert p.single_warp_safe is True
        assert p.sequential_halving_proven is True
        assert p.syncthreads_in_halving_loop is True

    def test_multi_warp_not_single_warp_safe(self):
        info = KernelParser.parse(REDUCTION_MULTI_WARP)
        p = info.reduction_proof
        # Has blockDim.x / 2 → not single-warp safe
        assert p is None or p.single_warp_safe is False

    def test_non_reduction_no_proof(self):
        info = KernelParser.parse(VEC_ADD)
        assert info.reduction_proof is None


class TestAlignmentProof:

    def test_scalar_loads_have_proof(self):
        info = KernelParser.parse(SCALAR_LOADS)
        assert info.alignment_proof is not None
        assert info.alignment_proof.valid is True

    def test_loads_proven_safe(self):
        info = KernelParser.parse(SCALAR_LOADS)
        p = info.alignment_proof
        assert len(p.loads) == 2
        for lp in p.loads:
            assert lp.has_bounds_guard is True
            assert lp.stride_is_one is True
            assert lp.no_pointer_cast is True

    def test_no_loop_no_proof(self):
        info = KernelParser.parse(VEC_ADD)
        # VEC_ADD has a loop, but let's check
        # Actually VEC_ADD does have bounds but no scalar load pattern
        assert info.alignment_proof is None


class TestCoarseningProof:

    def test_vec_add_coarsenable(self):
        info = KernelParser.parse(VEC_ADD)
        assert info.coarsening_proof is not None
        assert info.coarsening_proof.valid is True
        assert info.coarsening_proof.idx_var == "idx"
        assert info.coarsening_proof.no_atomics is True
        assert info.coarsening_proof.no_syncthreads is True
        assert info.coarsening_proof.no_shared_memory is True
        assert info.coarsening_proof.no_early_return is True

    def test_reduction_not_coarsenable(self):
        info = KernelParser.parse(REDUCTION)
        p = info.coarsening_proof
        # Has syncthreads and shared memory → invalid
        assert p is None or p.valid is False


class TestUnrollProof:

    def test_loops_have_unroll_proof(self):
        info = KernelParser.parse(VEC_ADD)
        p = info.unroll_proof
        assert p is not None
        assert p.valid is True
        assert p.no_existing_pragma is True

    def test_already_unrolled_no_proof(self):
        info = KernelParser.parse(UNROLLED)
        p = info.unroll_proof
        assert p is None or p.valid is False