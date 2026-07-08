"""Tests for RegisterBlocking transform."""

import pytest
from cuda_opt_agent.schemas.bottleneck import BottleneckClass, BottleneckResult
from cuda_opt_agent.schemas.kernel import KernelLanguage, KernelSource
from cuda_opt_agent.transforms.text.register_blocking import RegisterBlocking

MATMUL_SRC = """\
__global__ void matmul(float *C, float *A, float *B, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    for (int k = 0; k < N; k++) {
        sum += A[row * N + k] * B[k * N + col];
    }
    C[row * N + col] = sum;
}
"""

VECADD_SRC = """\
__global__ void vecadd(float *C, float *A, float *B, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) C[i] = A[i] + B[i];
}
"""

BOTTLENECK = BottleneckResult(primary=BottleneckClass.COMPUTE_BOUND, confidence=0.8)


class TestRegisterBlocking:

    @pytest.fixture
    def transform(self):
        return RegisterBlocking()

    def test_name(self, transform):
        assert transform.name == "register_blocking"

    def test_applicable(self, transform):
        kernel = KernelSource(name="test.cu", language=KernelLanguage.CUDA, source_code=MATMUL_SRC)
        assert transform.is_applicable(kernel, BOTTLENECK, {}) is True

    def test_not_applicable_vecadd(self, transform):
        kernel = KernelSource(name="test.cu", language=KernelLanguage.CUDA, source_code=VECADD_SRC)
        assert transform.is_applicable(kernel, BOTTLENECK, {}) is False

    def test_apply_2x2(self, transform):
        kernel = KernelSource(name="test.cu", language=KernelLanguage.CUDA, source_code=MATMUL_SRC)
        result = transform.apply(kernel, {"block_m": 2, "block_n": 2})
        assert result.success is True
        assert "sum_0_0" in result.transformed_source
        assert "sum_1_1" in result.transformed_source

    def test_apply_4x4(self, transform):
        kernel = KernelSource(name="test.cu", language=KernelLanguage.CUDA, source_code=MATMUL_SRC)
        result = transform.apply(kernel, {"block_m": 4, "block_n": 4})
        assert "sum_3_3" in result.transformed_source

    def test_validate(self, transform):
        kernel = KernelSource(name="test.cu", language=KernelLanguage.CUDA, source_code=MATMUL_SRC)
        result = transform.apply(kernel, {"block_m": 2, "block_n": 2})
        assert transform.validate(result) is True

    def test_warnings(self, transform):
        kernel = KernelSource(name="test.cu", language=KernelLanguage.CUDA, source_code=MATMUL_SRC)
        result = transform.apply(kernel, {"block_m": 2, "block_n": 2})
        assert any("2x2" in w for w in result.warnings)