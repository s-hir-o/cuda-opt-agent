"""Tests for WarpShuffleReduction transform."""

import pytest
from cuda_opt_agent.schemas.bottleneck import BottleneckClass, BottleneckResult
from cuda_opt_agent.schemas.kernel import KernelLanguage, KernelSource
from cuda_opt_agent.transforms.text.warp_shuffle import WarpShuffleReduction

REDUCE_KERNEL = """\
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

SIMPLE_KERNEL = """\
__global__ void add(float *C, float *A, float *B, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) C[i] = A[i] + B[i];
}
"""

BOTTLENECK = BottleneckResult(primary=BottleneckClass.MEMORY_BOUND, confidence=0.8)


class TestWarpShuffleReduction:

    @pytest.fixture
    def transform(self):
        return WarpShuffleReduction()

    def test_name(self, transform):
        assert transform.name == "warp_shuffle"

    def test_applicable_reduce(self, transform):
        kernel = KernelSource(name="test.cu", language=KernelLanguage.CUDA, source_code=REDUCE_KERNEL)
        assert transform.is_applicable(kernel, BOTTLENECK, {}) is True

    def test_not_applicable_simple(self, transform):
        kernel = KernelSource(name="test.cu", language=KernelLanguage.CUDA, source_code=SIMPLE_KERNEL)
        assert transform.is_applicable(kernel, BOTTLENECK, {}) is False

    def test_apply_inserts_shuffle(self, transform):
        kernel = KernelSource(name="test.cu", language=KernelLanguage.CUDA, source_code=REDUCE_KERNEL)
        result = transform.apply(kernel, {"reduction_op": "sum"})
        assert "__shfl_down_sync" in result.transformed_source
        assert "warp_reduce_sum" in result.transformed_source
        assert result.success is True

    def test_apply_preserves_output(self, transform):
        kernel = KernelSource(name="test.cu", language=KernelLanguage.CUDA, source_code=REDUCE_KERNEL)
        result = transform.apply(kernel, {"reduction_op": "sum"})
        assert "out[" in result.transformed_source
        assert "sdata" in result.transformed_source

    def test_validate(self, transform):
        kernel = KernelSource(name="test.cu", language=KernelLanguage.CUDA, source_code=REDUCE_KERNEL)
        result = transform.apply(kernel, {"reduction_op": "sum"})
        assert transform.validate(result) is True

    def test_no_comment_only_rewrite(self, transform):
        """Ensure the transform doesn't succeed with only a comment insertion."""
        kernel = KernelSource(name="test.cu", language=KernelLanguage.CUDA, source_code=REDUCE_KERNEL)
        result = transform.apply(kernel, {"reduction_op": "sum"})
        if result.success:
            import re
            from cuda_opt_agent.transforms.text.lexical import mask_comments_and_strings
            masked = mask_comments_and_strings(result.transformed_source)
            assert masked.filter_matches(re.compile(r"__shfl_down_sync"))