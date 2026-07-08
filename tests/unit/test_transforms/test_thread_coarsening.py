"""Tests for ThreadCoarsening transform."""

import pytest
from cuda_opt_agent.schemas.bottleneck import BottleneckClass, BottleneckResult
from cuda_opt_agent.schemas.kernel import KernelLanguage, KernelSource
from cuda_opt_agent.transforms.text.thread_coarsening import ThreadCoarsening

SIMPLE_KERNEL = """\
__global__ void add(float *C, float *A, float *B, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}
"""

ALREADY_GRID_STRIDE = """\
__global__ void add(float *C, float *A, float *B, int N) {
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < N;
         idx += gridDim.x * blockDim.x) {
        C[idx] = A[idx] + B[idx];
    }
}
"""

ATOMIC_KERNEL = """\
__global__ void hist(int *out, int *in, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        atomicAdd(&out[in[idx]], 1);
    }
}
"""

SYNC_KERNEL = """\
__global__ void reduce(float *out, float *in, int N) {
    __shared__ float sdata[256];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[threadIdx.x] = in[idx];
    __syncthreads();
    for (int s = 128; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) out[blockIdx.x] = sdata[0];
}
"""

BOTTLENECK = BottleneckResult(primary=BottleneckClass.MEMORY_BOUND, confidence=0.8)


class TestThreadCoarsening:

    @pytest.fixture
    def transform(self):
        return ThreadCoarsening()

    def test_name(self, transform):
        assert transform.name == "thread_coarsening"

    def test_applicable_simple(self, transform):
        kernel = KernelSource(name="test.cu", language=KernelLanguage.CUDA, source_code=SIMPLE_KERNEL)
        assert transform.is_applicable(kernel, BOTTLENECK, {}) is True

    def test_not_applicable_already_grid_stride(self, transform):
        kernel = KernelSource(name="test.cu", language=KernelLanguage.CUDA, source_code=ALREADY_GRID_STRIDE)
        assert transform.is_applicable(kernel, BOTTLENECK, {}) is False

    def test_not_applicable_atomic(self, transform):
        kernel = KernelSource(name="test.cu", language=KernelLanguage.CUDA, source_code=ATOMIC_KERNEL)
        assert transform.is_applicable(kernel, BOTTLENECK, {}) is False

    def test_not_applicable_sync(self, transform):
        kernel = KernelSource(name="test.cu", language=KernelLanguage.CUDA, source_code=SYNC_KERNEL)
        assert transform.is_applicable(kernel, BOTTLENECK, {}) is False

    def test_apply_adds_grid_stride(self, transform):
        kernel = KernelSource(name="test.cu", language=KernelLanguage.CUDA, source_code=SIMPLE_KERNEL)
        result = transform.apply(kernel, {"factor": 4})
        assert "gridDim" in result.transformed_source
        assert result.success is True

    def test_apply_factor_affects_output(self, transform):
        kernel = KernelSource(name="test.cu", language=KernelLanguage.CUDA, source_code=SIMPLE_KERNEL)
        result2 = transform.apply(kernel, {"factor": 2})
        result8 = transform.apply(kernel, {"factor": 8})
        assert "* 2" in result2.transformed_source
        assert "* 8" in result8.transformed_source

    def test_validate(self, transform):
        kernel = KernelSource(name="test.cu", language=KernelLanguage.CUDA, source_code=SIMPLE_KERNEL)
        result = transform.apply(kernel, {"factor": 4})
        assert transform.validate(result) is True