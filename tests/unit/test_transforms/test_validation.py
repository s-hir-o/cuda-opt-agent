"""Tests for the validation helper."""

import pytest
from cuda_opt_agent.transforms.validation import validate_transform


VALID_KERNEL = """\
__global__ void add(float *C, const float *A, const float *B, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}
"""

BALANCED_BRACES = "{ { } }"
UNBALANCED_OPEN = "{ { }"
UNBALANCED_CLOSE = "{ } }"


class TestValidation:

    def test_valid_kernel_passes(self):
        assert validate_transform(VALID_KERNEL) is True

    def test_balanced_braces_pass(self):
        assert validate_transform(BALANCED_BRACES + "\n__global__ void k() { }") is True

    def test_unbalanced_open_fails(self):
        assert validate_transform(UNBALANCED_OPEN) is False

    def test_unbalanced_close_fails(self):
        assert validate_transform(UNBALANCED_CLOSE) is False

    def test_no_kernel_fails(self):
        assert validate_transform("int x = 1;") is False

    def test_empty_body_fails(self):
        assert validate_transform("__global__ void k() { }") is False

    def test_placeholder_base_fails(self):
        src = "__global__ void k() { float val = input_base[0]; }"
        assert validate_transform(src) is False

    def test_placeholder_tmp_var_fails(self):
        src = "__global__ void k() { int TMP_VAR_x = 1; }"
        assert validate_transform(src) is False

    def test_unclosed_string_fails(self):
        src = '__global__ void k() { printf("unterminated); }'
        assert validate_transform(src) is False

    def test_comment_preservation_check(self):
        orig = "/* comment */ __global__ void k() { int x = 1; }"
        transformed = "/* CHANGED */ __global__ void k() { int x = 1; }"
        assert validate_transform(transformed, original_source=orig) is False

    def test_balanced_parentheses_pass(self):
        src = "__global__ void k() { int x = (1 + (2 * 3)); }"
        assert validate_transform(src) is True

    def test_unbalanced_paren_fails(self):
        src = "__global__ void k() { int x = (1 + (2 * 3); }"
        assert validate_transform(src) is False

    def test_balanced_brackets_pass(self):
        src = "__global__ void k() { float a[10]; a[0] = 1.0f; }"
        assert validate_transform(src) is True