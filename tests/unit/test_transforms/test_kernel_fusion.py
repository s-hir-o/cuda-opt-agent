"""Tests for KernelFusion transform."""

import pytest
from cuda_opt_agent.schemas.bottleneck import BottleneckClass, BottleneckResult
from cuda_opt_agent.schemas.kernel import KernelLanguage, KernelSource
from cuda_opt_agent.transforms.text.kernel_fusion import KernelFusion

PRODUCER_CONSUMER = """\
__global__ void scale(float *out, const float *in, float s, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) out[idx] = in[idx] * s;
}

__global__ void add(float *C, const float *A, const float *B, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) C[idx] = A[idx] + B[idx];
}
"""

NO_DEPENDENCY = """\
__global__ void scale(float *X, const float *in, float s, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) X[idx] = in[idx] * s;
}

__global__ void add(float *Y, const float *A, const float *B, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) Y[idx] = A[idx] + B[idx];
}
"""

SINGLE_KERNEL = """\
__global__ void add(float *C, const float *A, const float *B, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) C[idx] = A[idx] + B[idx];
}
"""

BOTTLENECK = BottleneckResult(primary=BottleneckClass.MEMORY_BOUND, confidence=0.8)


class TestKernelFusion:

    @pytest.fixture
    def transform(self):
        return KernelFusion()

    def test_name(self, transform):
        assert transform.name == "kernel_fusion"

    def test_applicable_producer_consumer(self, transform):
        kernel = KernelSource(name="test.cu", language=KernelLanguage.CUDA, source_code=PRODUCER_CONSUMER)
        assert transform.is_applicable(kernel, BOTTLENECK, {}) is True

    def test_not_applicable_single_kernel(self, transform):
        kernel = KernelSource(name="test.cu", language=KernelLanguage.CUDA, source_code=SINGLE_KERNEL)
        assert transform.is_applicable(kernel, BOTTLENECK, {}) is False

    def test_not_applicable_no_dependency(self, transform):
        kernel = KernelSource(name="test.cu", language=KernelLanguage.CUDA, source_code=NO_DEPENDENCY)
        # These share no output/read buffer dependency
        assert transform.is_applicable(kernel, BOTTLENECK, {}) is False

    def test_apply_produces_fused(self, transform):
        kernel = KernelSource(name="test.cu", language=KernelLanguage.CUDA, source_code=PRODUCER_CONSUMER)
        result = transform.apply(kernel, {"fused_kernel_name": "fused"})
        assert "fused" in result.transformed_source.lower()
        assert result.success is True

    def test_validate(self, transform):
        kernel = KernelSource(name="test.cu", language=KernelLanguage.CUDA, source_code=PRODUCER_CONSUMER)
        result = transform.apply(kernel, {})
        assert transform.validate(result) is True

    def test_preserves_original(self, transform):
        kernel = KernelSource(name="test.cu", language=KernelLanguage.CUDA, source_code=PRODUCER_CONSUMER)
        result = transform.apply(kernel, {})
        assert "scale" in result.transformed_source
        assert "add" in result.transformed_source