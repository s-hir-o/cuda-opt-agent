"""Tests for VectorizedLoads transform."""

import pytest
from cuda_opt_agent.schemas.bottleneck import BottleneckClass, BottleneckResult
from cuda_opt_agent.schemas.kernel import KernelLanguage, KernelSource
from cuda_opt_agent.transforms.text.vectorized_loads import VectorizedLoads

SCALAR_LOAD_KERNEL = """\
__global__ void scale(float *out, float *in, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float val = in[idx];
        out[idx] = val * 2.0f;
    }
}
"""

NO_LOAD_KERNEL = """\
__global__ void noop(int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
}
"""

NO_BOUNDS_GUARD = """\
__global__ void unsafe(float *out, float *in, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float val = in[idx];
    out[idx] = val;
}
"""

BOTTLENECK = BottleneckResult(primary=BottleneckClass.MEMORY_BOUND, confidence=0.9)


class TestVectorizedLoads:

    @pytest.fixture
    def transform(self):
        return VectorizedLoads()

    def test_name(self, transform):
        assert transform.name == "vectorized_loads"

    def test_applicable(self, transform):
        kernel = KernelSource(name="test.cu", language=KernelLanguage.CUDA, source_code=SCALAR_LOAD_KERNEL)
        assert transform.is_applicable(kernel, BOTTLENECK, {}) is True

    def test_not_applicable_no_load(self, transform):
        kernel = KernelSource(name="test.cu", language=KernelLanguage.CUDA, source_code=NO_LOAD_KERNEL)
        assert transform.is_applicable(kernel, BOTTLENECK, {}) is False

    def test_not_applicable_no_bounds_guard(self, transform):
        kernel = KernelSource(name="test.cu", language=KernelLanguage.CUDA, source_code=NO_BOUNDS_GUARD)
        assert transform.is_applicable(kernel, BOTTLENECK, {}) is False

    def test_apply_adds_float4(self, transform):
        kernel = KernelSource(name="test.cu", language=KernelLanguage.CUDA, source_code=SCALAR_LOAD_KERNEL)
        result = transform.apply(kernel, {"vector_width": 4})
        assert "float4" in result.transformed_source
        assert result.success is True

    def test_apply_width_2(self, transform):
        kernel = KernelSource(name="test.cu", language=KernelLanguage.CUDA, source_code=SCALAR_LOAD_KERNEL)
        result = transform.apply(kernel, {"vector_width": 2})
        assert "float2" in result.transformed_source

    def test_validate(self, transform):
        kernel = KernelSource(name="test.cu", language=KernelLanguage.CUDA, source_code=SCALAR_LOAD_KERNEL)
        result = transform.apply(kernel, {"vector_width": 4})
        assert transform.validate(result) is True

    def test_no_invented_symbols(self, transform):
        kernel = KernelSource(name="test.cu", language=KernelLanguage.CUDA, source_code=SCALAR_LOAD_KERNEL)
        result = transform.apply(kernel, {"vector_width": 4})
        if result.success:
            assert "_base" not in result.transformed_source