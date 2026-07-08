"""Tests for structural validation."""

import pytest

from cuda_opt_agent.transforms.validation import validate_transform


class TestValidation:

    def test_balanced_pass(self):
        assert validate_transform(
            '__global__ void k(float *a) { a[0] = 1.0f; }'
        ) is True

    def test_unclosed_brace(self):
        assert validate_transform(
            '__global__ void k(float *a) { a[0] = 1.0f;'
        ) is False

    def test_extra_brace(self):
        assert validate_transform(
            '__global__ void k(float *a) } a[0] = 1.0f; {'
        ) is False

    def test_unclosed_paren(self):
        assert validate_transform(
            '__global__ void k(float *a) { float x = (1.0f + 2.0f; }'
        ) is False

    def test_unclosed_bracket(self):
        assert validate_transform(
            '__global__ void k(float *a) { float x = a[0; }'
        ) is False

    def test_no_kernel(self):
        assert validate_transform('void foo() { int x = 1; }') is False

    def test_empty_body(self):
        assert validate_transform('__global__ void k(float *a) {}') is False

    def test_placeholder_base(self):
        assert validate_transform(
            '__global__ void k(float *a) { a[0] = result_base; }'
        ) is False

    def test_placeholder_tmp(self):
        assert validate_transform(
            '__global__ void k(float *a) { float x = TMP_VAR_0; }'
        ) is False

    def test_comment_preserved(self):
        orig = '/* hi */\n__global__ void k(float *a) { a[0] = 1.0f; }'
        trans = '/* hi */\n__global__ void k(float *a) { a[0] = 2.0f; }'
        assert validate_transform(trans, original_source=orig) is True

    def test_comment_mutated(self):
        orig = '/* hi */\n__global__ void k(float *a) { a[0] = 1.0f; }'
        trans = '/* bye */\n__global__ void k(float *a) { a[0] = 2.0f; }'
        assert validate_transform(trans, original_source=orig) is False

    def test_string_preserved(self):
        orig = '__global__ void k(float *a) { printf("hi"); a[0] = 1.0f; }'
        trans = '__global__ void k(float *a) { printf("hi"); a[0] = 2.0f; }'
        assert validate_transform(trans, original_source=orig) is True

    def test_string_mutated(self):
        orig = '__global__ void k(float *a) { printf("hi"); a[0] = 1.0f; }'
        trans = '__global__ void k(float *a) { printf("bye"); a[0] = 2.0f; }'
        assert validate_transform(trans, original_source=orig) is False

    def test_braces_in_comment_ok(self):
        assert validate_transform(
            '/* { */\n__global__ void k(float *a) { a[0] = 1.0f; }'
        ) is True

    def test_check_comments_false(self):
        orig = '/* hi */\n__global__ void k(float *a) { a[0] = 1.0f; }'
        trans = '__global__ void k(float *a) { a[0] = 2.0f; }'
        assert validate_transform(
            trans, original_source=orig, check_comments=False
        ) is True

    def test_result_success_on_pass(self):
        from cuda_opt_agent.schemas.transform import TransformResult, TransformSpec, TransformBackend
        spec = TransformSpec(
            transform_name="test", backend=TransformBackend.TEXT,
            params={}, reason=""
        )
        r = TransformResult(
            spec=spec, original_source="",
            transformed_source='__global__ void k(float *a) { a[0] = 1.0f; }',
            success=False,
        )
        validate_transform(r.transformed_source, result=r)
        assert r.success is True

    def test_result_cleared_on_fail(self):
        from cuda_opt_agent.schemas.transform import TransformResult, TransformSpec, TransformBackend
        spec = TransformSpec(
            transform_name="test", backend=TransformBackend.TEXT,
            params={}, reason=""
        )
        r = TransformResult(
            spec=spec, original_source="",
            transformed_source='__global__ void k(float *a) {',
            success=True,
        )
        validate_transform(r.transformed_source, result=r)
        assert r.success is False