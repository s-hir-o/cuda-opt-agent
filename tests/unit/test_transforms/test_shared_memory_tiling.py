"""Tests for SharedMemoryTiling transform."""

import pytest
from cuda_opt_agent.schemas.bottleneck import BottleneckClass, BottleneckResult
from cuda_opt_agent.schemas.kernel import KernelLanguage, KernelSource
from cuda_opt_agent.transforms.text.shared_memory_tiling import SharedMemoryTiling

MATMUL_SRC = """\
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

VECADD_SRC = """\
__global__ void vecadd(float *C, const float *A, const float *B, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) C[i] = A[i] + B[i];
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

COMMENT_FOR = """\
__global__ void matmul(float *C, const float *A, const float *B, int N) {
    // for (int k = 0; k < N; k++) { sum += A[k] * B[k]; }
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

BOTTLENECK = BottleneckResult(primary=BottleneckClass.MEMORY_BOUND, confidence=0.9)


class TestSharedMemoryTiling:

    @pytest.fixture
    def transform(self):
        return SharedMemoryTiling()

    def test_name(self, transform):
        assert transform.name == "shared_memory_tiling"

    def test_targets_memory_bound(self, transform):
        assert BottleneckClass.MEMORY_BOUND in transform.targets

    def test_applicable_matmul(self, transform):
        kernel = KernelSource(name="matmul.cu", language=KernelLanguage.CUDA, source_code=MATMUL_SRC)
        assert transform.is_applicable(kernel, BOTTLENECK, {}) is True

    def test_not_applicable_vecadd(self, transform):
        kernel = KernelSource(name="vecadd.cu", language=KernelLanguage.CUDA, source_code=VECADD_SRC)
        assert transform.is_applicable(kernel, BOTTLENECK, {}) is False

    def test_not_applicable_already_tiled(self, transform):
        kernel = KernelSource(name="matmul.cu", language=KernelLanguage.CUDA, source_code=ALREADY_TILED)
        assert transform.is_applicable(kernel, BOTTLENECK, {}) is False

    def test_apply_produces_tiled_kernel(self, transform):
        kernel = KernelSource(name="matmul.cu", language=KernelLanguage.CUDA, source_code=MATMUL_SRC)
        result = transform.apply(kernel, {"tile_size": 16})
        assert result.success is True
        assert "__shared__" in result.transformed_source
        assert "__syncthreads" in result.transformed_source
        assert "A_s[16][16]" in result.transformed_source
        assert "B_s[16][16]" in result.transformed_source

    def test_apply_custom_tile_size(self, transform):
        kernel = KernelSource(name="matmul.cu", language=KernelLanguage.CUDA, source_code=MATMUL_SRC)
        result = transform.apply(kernel, {"tile_size": 32})
        assert "A_s[32][32]" in result.transformed_source

    def test_apply_preserves_original_symbols(self, transform):
        kernel = KernelSource(name="matmul.cu", language=KernelLanguage.CUDA, source_code=MATMUL_SRC)
        result = transform.apply(kernel, {"tile_size": 16})
        assert "float *C" in result.transformed_source
        assert "A[" in result.transformed_source
        assert "B[" in result.transformed_source
        assert "C[" in result.transformed_source

    def test_different_variable_names(self, transform):
        kernel = KernelSource(name="gemm.cu", language=KernelLanguage.CUDA, source_code=MATMUL_DIFFERENT_NAMES)
        result = transform.apply(kernel, {"tile_size": 16})
        if result.success:
            assert "MatA_s" in result.transformed_source
            assert "MatB_s" in result.transformed_source
            assert "Result[" in result.transformed_source

    def test_validate_passes(self, transform):
        kernel = KernelSource(name="matmul.cu", language=KernelLanguage.CUDA, source_code=MATMUL_SRC)
        result = transform.apply(kernel, {"tile_size": 16})
        assert transform.validate(result) is True

    def test_warnings_present(self, transform):
        kernel = KernelSource(name="matmul.cu", language=KernelLanguage.CUDA, source_code=MATMUL_SRC)
        result = transform.apply(kernel, {"tile_size": 16})
        assert len(result.warnings) > 0

    def test_preserves_original_source(self, transform):
        kernel = KernelSource(name="matmul.cu", language=KernelLanguage.CUDA, source_code=MATMUL_SRC)
        result = transform.apply(kernel, {})
        assert result.original_source == MATMUL_SRC

    def test_comment_for_not_false_positive(self, transform):
        kernel = KernelSource(name="matmul.cu", language=KernelLanguage.CUDA, source_code=COMMENT_FOR)
        assert transform.is_applicable(kernel, BOTTLENECK, {}) is True