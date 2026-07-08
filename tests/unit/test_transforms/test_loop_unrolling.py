"""Tests for LoopUnrolling transform."""

import pytest
from cuda_opt_agent.schemas.bottleneck import BottleneckClass, BottleneckResult
from cuda_opt_agent.schemas.kernel import KernelLanguage, KernelSource
from cuda_opt_agent.transforms.text.loop_unrolling import LoopUnrolling

KERNEL_WITH_LOOP = """\
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

KERNEL_NO_LOOP = """\
__global__ void scale(float *out, const float *in, float s, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) out[idx] = in[idx] * s;
}
"""

KERNEL_ALREADY_UNROLLED = """\
__global__ void compute(float *out, const float *in, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    #pragma unroll 4
    for (int k = 0; k < 8; k++) {
        out[idx * 8 + k] = in[idx * 8 + k];
    }
}
"""

COMMENT_LOOP = """\
__global__ void compute(float *out, const float *in, int N) {
    // for (int i = 0; i < 100; i++) { do_something(); }
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

BOTTLENECK = BottleneckResult(primary=BottleneckClass.COMPUTE_BOUND, confidence=0.8)


class TestLoopUnrolling:

    @pytest.fixture
    def transform(self):
        return LoopUnrolling()

    def test_applicable_with_loop(self, transform):
        kernel = KernelSource(name="test.cu", language=KernelLanguage.CUDA, source_code=KERNEL_WITH_LOOP)
        assert transform.is_applicable(kernel, BOTTLENECK, {}) is True

    def test_not_applicable_no_loop(self, transform):
        kernel = KernelSource(name="test.cu", language=KernelLanguage.CUDA, source_code=KERNEL_NO_LOOP)
        assert transform.is_applicable(kernel, BOTTLENECK, {}) is False

    def test_not_applicable_already_unrolled(self, transform):
        kernel = KernelSource(name="test.cu", language=KernelLanguage.CUDA, source_code=KERNEL_ALREADY_UNROLLED)
        assert transform.is_applicable(kernel, BOTTLENECK, {}) is False

    def test_apply_adds_pragma(self, transform):
        kernel = KernelSource(name="test.cu", language=KernelLanguage.CUDA, source_code=KERNEL_WITH_LOOP)
        result = transform.apply(kernel, {"factor": 4})
        assert "#pragma unroll" in result.transformed_source
        assert result.success is True

    def test_small_loop_full_unroll(self, transform):
        kernel = KernelSource(name="test.cu", language=KernelLanguage.CUDA, source_code=KERNEL_WITH_LOOP)
        result = transform.apply(kernel, {"factor": 4})
        assert "#pragma unroll 8" in result.transformed_source

    def test_validate(self, transform):
        kernel = KernelSource(name="test.cu", language=KernelLanguage.CUDA, source_code=KERNEL_WITH_LOOP)
        result = transform.apply(kernel, {"factor": 4})
        assert transform.validate(result) is True

    def test_targets(self, transform):
        assert BottleneckClass.COMPUTE_BOUND in transform.targets
        assert BottleneckClass.LATENCY_BOUND in transform.targets

    def test_comment_loop_not_targeted(self, transform):
        kernel = KernelSource(name="test.cu", language=KernelLanguage.CUDA, source_code=COMMENT_LOOP)
        result = transform.apply(kernel, {"factor": 4})
        # Should only add pragma to the real for-loop, not the comment
        assert result.success is True
        assert result.transformed_source.count("#pragma unroll") == 1