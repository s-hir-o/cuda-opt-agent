"""Integration tests for every concrete transform: proof-based applicability,
rejection, and round-trip validation.
"""

import pytest
import re

from cuda_opt_agent.schemas.bottleneck import BottleneckClass, BottleneckResult
from cuda_opt_agent.schemas.kernel import KernelSource
from cuda_opt_agent.transforms.registry import get_transform


# FIX: KernelSource requires `name`; was missing, causing 34 ValidationError
def _ks(source: str) -> KernelSource:
    return KernelSource(name="test.cu", source_code=source)


# FIX: field is `primary` not `bottleneck_class`; `confidence` is required
def _bn(cls: BottleneckClass = BottleneckClass.MEMORY_BOUND) -> BottleneckResult:
    return BottleneckResult(primary=cls, confidence=0.8)


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

MATMUL_SHARED = """\
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

VEC_ADD = """\
__global__ void vector_add(float *out, float *a, float *b, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        out[idx] = a[idx] + b[idx];
    }
}
"""

LOOP_KERNEL = """\
__global__ void scale(float *out, float *in, float factor, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float val = 0.0f;
        for (int i = 0; i < 4; i++) {
            val += in[idx * 4 + i];
        }
        out[idx] = val * factor;
    }
}
"""

ALREADY_UNROLLED = """\
__global__ void scale(float *out, float *in, float factor, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float val = 0.0f;
        #pragma unroll 4
        for (int i = 0; i < 4; i++) {
            val += in[idx * 4 + i];
        }
        out[idx] = val * factor;
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

TWO_KERNELS = """\
__global__ void kernel1(float *out, float *in, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        out[idx] = in[idx] * 2.0f;
    }
}

__global__ void kernel2(float *out, float *in, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        out[idx] = in[idx] + 1.0f;
    }
}
"""

TWO_UNRELATED = """\
__global__ void kernelA(float *a, float *x, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        a[idx] = x[idx] * 3.0f;
    }
}

__global__ void kernelB(float *b, float *y, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        b[idx] = y[idx] + 5.0f;
    }
}
"""


# ── shared_memory_tiling ────────────────────────────────────────────────────

class TestSharedMemoryTiling:

    def test_positive_matmul(self):
        t = get_transform("shared_memory_tiling")
        assert t.is_applicable(_ks(MATMUL), _bn(), {}) is True

    def test_negative_already_shared(self):
        t = get_transform("shared_memory_tiling")
        assert t.is_applicable(_ks(MATMUL_SHARED), _bn(), {}) is False

    def test_negative_not_matmul(self):
        t = get_transform("shared_memory_tiling")
        assert t.is_applicable(_ks(VEC_ADD), _bn(), {}) is False

    def test_negative_no_bounds_var(self):
        """Matmul without an int N parameter → no proven dimension → reject."""
        no_bounds = """\
__global__ void matmul(float *C, float *A, float *B) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    for (int k = 0; k < 1024; k++) {
        sum += A[row * 1024 + k] * B[k * 1024 + col];
    }
    C[row * 1024 + col] = sum;
}
"""
        t = get_transform("shared_memory_tiling")
        assert t.is_applicable(_ks(no_bounds), _bn(), {}) is False

    def test_validate_rejects_unbalanced(self):
        from cuda_opt_agent.schemas.transform import TransformResult, TransformSpec, TransformBackend
        spec = TransformSpec(
            transform_name="shared_memory_tiling",
            backend=TransformBackend.TEXT, params={}, reason=""
        )
        result = TransformResult(
            spec=spec, original_source=MATMUL,
            transformed_source="__global__ void k() {", success=False
        )
        t = get_transform("shared_memory_tiling")
        assert t.validate(result) is False


# ── register_blocking ───────────────────────────────────────────────────────

class TestRegisterBlocking:

    def test_positive_matmul(self):
        t = get_transform("register_blocking")
        assert t.is_applicable(_ks(MATMUL), _bn(), {}) is True

    def test_negative_not_matmul(self):
        t = get_transform("register_blocking")
        assert t.is_applicable(_ks(VEC_ADD), _bn(), {}) is False

    def test_negative_already_shared(self):
        t = get_transform("register_blocking")
        assert t.is_applicable(_ks(MATMUL_SHARED), _bn(), {}) is False


# ── loop_unrolling ──────────────────────────────────────────────────────────

class TestLoopUnrolling:

    def test_positive(self):
        t = get_transform("loop_unrolling")
        assert t.is_applicable(_ks(LOOP_KERNEL), _bn(), {}) is True

    def test_negative_already_unrolled(self):
        t = get_transform("loop_unrolling")
        assert t.is_applicable(_ks(ALREADY_UNROLLED), _bn(), {}) is False

    def test_negative_no_loop(self):
        t = get_transform("loop_unrolling")
        assert t.is_applicable(_ks(VEC_ADD), _bn(), {}) is False

    def test_round_trip(self):
        t = get_transform("loop_unrolling")
        result = t.apply(_ks(LOOP_KERNEL), {})
        assert t.validate(result) is True
        assert result.success is True
        assert "#pragma unroll" in result.transformed_source


# ── vectorized_loads ────────────────────────────────────────────────────────

class TestVectorizedLoads:

    def test_positive(self):
        t = get_transform("vectorized_loads")
        assert t.is_applicable(_ks(SCALAR_LOADS), _bn(), {}) is True

    def test_negative_no_bounds_guard(self):
        no_guard = """\
__global__ void k(float *out, float *a, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float va = a[idx];
    out[idx] = va;
}
"""
        t = get_transform("vectorized_loads")
        assert t.is_applicable(_ks(no_guard), _bn(), {}) is False

    def test_negative_no_loop(self):
        t = get_transform("vectorized_loads")
        assert t.is_applicable(_ks(VEC_ADD), _bn(), {}) is False

    def test_round_trip(self):
        t = get_transform("vectorized_loads")
        result = t.apply(_ks(SCALAR_LOADS), {"vector_width": 4})
        assert t.validate(result) is True
        assert "float4" in result.transformed_source
        assert result.success is True

    def test_no_placeholder_symbols(self):
        t = get_transform("vectorized_loads")
        result = t.apply(_ks(SCALAR_LOADS), {"vector_width": 4})
        assert t.validate(result) is True
        assert not re.search(r"\bplaceholder_\w+\b", result.transformed_source)
        assert not re.search(r"\bTMP_VAR_\w+\b", result.transformed_source)

    def test_negative_pointer_casts(self):
        casts = """\
__global__ void k(float *out, float *a, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float4 v = *(float4*)(&a[idx * 4]);
        out[idx] = v.x;
    }
}
"""
        t = get_transform("vectorized_loads")
        # Already has pointer casts → AlignmentProof has no_pointer_cast=False
        assert t.is_applicable(_ks(casts), _bn(), {}) is False


# ── thread_coarsening ───────────────────────────────────────────────────────

class TestThreadCoarsening:

    def test_positive(self):
        t = get_transform("thread_coarsening")
        assert t.is_applicable(_ks(VEC_ADD), _bn(), {}) is True

    def test_negative_syncthreads(self):
        t = get_transform("thread_coarsening")
        assert t.is_applicable(_ks(REDUCTION), _bn(), {}) is False

    def test_negative_shared_memory(self):
        t = get_transform("thread_coarsening")
        assert t.is_applicable(_ks(MATMUL_SHARED), _bn(), {}) is False

    def test_negative_early_return(self):
        early_ret = """\
__global__ void k(float *out, float *a, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    out[idx] = a[idx] * 2.0f;
}
"""
        t = get_transform("thread_coarsening")
        assert t.is_applicable(_ks(early_ret), _bn(), {}) is False

    def test_round_trip(self):
        t = get_transform("thread_coarsening")
        result = t.apply(_ks(VEC_ADD), {"factor": 4})
        assert t.validate(result) is True
        assert "gridDim" in result.transformed_source
        assert result.success is True


# ── warp_shuffle ────────────────────────────────────────────────────────────

class TestWarpShuffle:

    def test_positive_single_warp(self):
        t = get_transform("warp_shuffle")
        assert t.is_applicable(_ks(REDUCTION), _bn(), {}) is True

    def test_negative_multi_warp(self):
        t = get_transform("warp_shuffle")
        assert t.is_applicable(_ks(REDUCTION_MULTI_WARP), _bn(), {}) is False

    def test_negative_not_reduction(self):
        t = get_transform("warp_shuffle")
        assert t.is_applicable(_ks(VEC_ADD), _bn(), {}) is False

    def test_round_trip(self):
        t = get_transform("warp_shuffle")
        result = t.apply(_ks(REDUCTION), {})
        assert t.validate(result) is True
        assert "__shfl_down_sync" in result.transformed_source
        assert "warp_reduce" in result.transformed_source
        assert result.success is True

    def test_output_buffer_preserved(self):
        t = get_transform("warp_shuffle")
        result = t.apply(_ks(REDUCTION), {})
        assert "out" in result.transformed_source


# ── kernel_fusion ───────────────────────────────────────────────────────────

class TestKernelFusion:

    def test_positive_producer_consumer(self):
        t = get_transform("kernel_fusion")
        assert t.is_applicable(_ks(TWO_KERNELS), _bn(), {}) is True

    def test_negative_unrelated(self):
        t = get_transform("kernel_fusion")
        assert t.is_applicable(_ks(TWO_UNRELATED), _bn(), {}) is False

    def test_negative_single_kernel(self):
        t = get_transform("kernel_fusion")
        assert t.is_applicable(_ks(VEC_ADD), _bn(), {}) is False

    def test_round_trip(self):
        t = get_transform("kernel_fusion")
        result = t.apply(_ks(TWO_KERNELS), {})
        assert t.validate(result) is True
        assert "fused_kernel" in result.transformed_source
        assert result.success is True

    def test_no_original_in_comment(self):
        """Original source must NOT appear as a comment in output."""
        t = get_transform("kernel_fusion")
        result = t.apply(_ks(TWO_KERNELS), {})
        assert "ORIGINAL KERNELS" not in result.transformed_source


# ── _make_result safety ─────────────────────────────────────────────────────

class TestMakeResultSafety:

    def test_make_result_defaults_false(self):
        t = get_transform("loop_unrolling")
        result = t.apply(_ks(LOOP_KERNEL), {})
        # After apply + validate, success should be True
        assert t.validate(result) is True
        assert result.success is True

    def test_unvalidated_is_false(self):
        """Direct _make_result should produce success=False."""
        from cuda_opt_agent.schemas.transform import TransformResult
        t = get_transform("loop_unrolling")
        raw_result = t._make_result(_ks(LOOP_KERNEL), "garbage", {})
        assert raw_result.success is False