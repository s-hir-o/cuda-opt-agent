"""Tests for the structural parser (analysis.py)."""

import pytest
from cuda_opt_agent.transforms.analysis import KernelParser

CANONICAL_MATMUL = """\
__global__ void matmul(float *C, const float *A, const float *B, int N) {
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

VECADD = """\
__global__ void vecadd(float *C, const float *A, const float *B, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}
"""

REDUCTION = """\
__global__ void reduce(float *out, float *in, int N) {
    __shared__ float sdata[256];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[threadIdx.x] = (idx < N) ? in[idx] : 0.0f;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }

    if (threadIdx.x == 0) out[blockIdx.x] = sdata[0];
}
"""

SIMPLE_LOOP = """\
__global__ void compute(float *out, const float *in, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float sum = 0.0f;
        for (int k = 0; k < 8; k++) {
            sum += in[idx * 8 + k];
        }
        out[idx] = sum;
    }
}
"""

MATMUL_DIFFERENT_NAMES = """\
__global__ void gemm(float *Result, const float *MatA, const float *MatB, int Dim) {
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (r < Dim && c < Dim) {
        float acc = 0.0f;
        for (int p = 0; p < Dim; p++) {
            acc += MatA[r * Dim + p] * MatB[p * Dim + c];
        }
        Result[r * Dim + c] = acc;
    }
}
"""

ALREADY_TILED = """\
__global__ void matmul(float *C, const float *A, const float *B, int N) {
    __shared__ float As[16][16];
    __shared__ float Bs[16][16];
    int row = blockIdx.y * 16 + threadIdx.y;
    int col = blockIdx.x * 16 + threadIdx.x;
    float sum = 0.0f;
    for (int t = 0; t < N; t += 16) {
        As[threadIdx.y][threadIdx.x] = A[row * N + t + threadIdx.x];
        Bs[threadIdx.y][threadIdx.x] = B[(t + threadIdx.y) * N + col];
        __syncthreads();
        for (int k = 0; k < 16; k++) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        __syncthreads();
    }
    if (row < N && col < N)
        C[row * N + col] = sum;
}
"""


class TestKernelParser:

    def test_matmul_name(self):
        info = KernelParser.parse(CANONICAL_MATMUL)
        assert info.name == "matmul"

    def test_matmul_is_matmul_like(self):
        info = KernelParser.parse(CANONICAL_MATMUL)
        assert info.is_matmul_like is True

    def test_matmul_accumulation_variable(self):
        info = KernelParser.parse(CANONICAL_MATMUL)
        assert info.accumulation_variable == "sum"

    def test_matmul_read_buffers(self):
        info = KernelParser.parse(CANONICAL_MATMUL)
        assert "A" in info.read_buffer_names
        assert "B" in info.read_buffer_names

    def test_matmul_output_buffer(self):
        info = KernelParser.parse(CANONICAL_MATMUL)
        assert info.output_write_buffer == "C"

    def test_matmul_has_loop(self):
        info = KernelParser.parse(CANONICAL_MATMUL)
        assert info.has_loop is True

    def test_matmul_parameters(self):
        info = KernelParser.parse(CANONICAL_MATMUL)
        names = [p.name for p in info.parameters]
        assert "C" in names
        assert "A" in names
        assert "B" in names
        assert "N" in names

    def test_matmul_pointer_params(self):
        info = KernelParser.parse(CANONICAL_MATMUL)
        pointers = info.get_pointer_params()
        pointer_names = [p.name for p in pointers]
        assert "C" in pointer_names
        assert "A" in pointer_names
        assert "B" in pointer_names

    def test_matmul_not_reduction(self):
        info = KernelParser.parse(CANONICAL_MATMUL)
        assert info.is_reduction_like is False

    # --- vecadd ---

    def test_vecadd_not_matmul(self):
        info = KernelParser.parse(VECADD)
        assert info.is_matmul_like is False

    def test_vecadd_not_reduction(self):
        info = KernelParser.parse(VECADD)
        assert info.is_reduction_like is False

    def test_vecadd_has_loop(self):
        info = KernelParser.parse(VECADD)
        assert info.has_loop is False  # No for-loop

    # --- reduction ---

    def test_reduction_is_reduction_like(self):
        info = KernelParser.parse(REDUCTION)
        assert info.is_reduction_like is True

    def test_reduction_not_matmul(self):
        info = KernelParser.parse(REDUCTION)
        assert info.is_matmul_like is False

    def test_reduction_has_shared_memory(self):
        info = KernelParser.parse(REDUCTION)
        assert info.has_shared_memory is True

    def test_reduction_has_syncthreads(self):
        info = KernelParser.parse(REDUCTION)
        assert info.has_syncthreads is True

    def test_reduction_shared_buffers(self):
        info = KernelParser.parse(REDUCTION)
        assert "sdata" in info.shared_memory_buffers

    # --- simple loop ---

    def test_simple_loop_has_loop(self):
        info = KernelParser.parse(SIMPLE_LOOP)
        assert info.has_loop is True

    def test_simple_loop_not_matmul(self):
        info = KernelParser.parse(SIMPLE_LOOP)
        assert info.is_matmul_like is False

    # --- different variable names ---

    def test_different_names_is_matmul(self):
        info = KernelParser.parse(MATMUL_DIFFERENT_NAMES)
        assert info.is_matmul_like is True

    def test_different_names_accumulation(self):
        info = KernelParser.parse(MATMUL_DIFFERENT_NAMES)
        assert info.accumulation_variable == "acc"

    def test_different_names_buffers(self):
        info = KernelParser.parse(MATMUL_DIFFERENT_NAMES)
        assert "MatA" in info.read_buffer_names
        assert "MatB" in info.read_buffer_names
        assert info.output_write_buffer == "Result"

    # --- already tiled ---

    def test_already_tiled_has_shared_memory(self):
        info = KernelParser.parse(ALREADY_TILED)
        assert info.has_shared_memory is True

    def test_already_tiled_is_matmul(self):
        info = KernelParser.parse(ALREADY_TILED)
        assert info.is_matmul_like is True

    # --- empty/malformed ---

    def test_empty_source(self):
        info = KernelParser.parse("")
        assert info.name == ""

    def test_no_kernel(self):
        info = KernelParser.parse("int main() { return 0; }")
        assert info.name == ""

    def test_incomplete_kernel(self):
        info = KernelParser.parse("__global__ void foo(")
        assert info.name == "foo"
        assert info.body == ""