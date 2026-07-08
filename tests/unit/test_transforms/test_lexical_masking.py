"""Tests for the lexical masking layer."""

import pytest
from cuda_opt_agent.transforms.text.lexical import (
    mask_comments_and_strings,
    code_spans_contain,
)


class TestMaskCommentsAndStrings:

    def test_no_comments_or_strings(self):
        src = "int idx = blockIdx.x * blockDim.x + threadIdx.x;"
        masked = mask_comments_and_strings(src)
        assert masked.source == src
        assert len(masked.code_ranges) == 1
        assert masked.code_ranges[0] == (0, len(src))

    def test_line_comment(self):
        src = "int x = 1; // this is a comment\nint y = 2;"
        masked = mask_comments_and_strings(src)
        # 'int x = 1; ' is code, '// this...' is comment, '\nint y = 2;' is code
        assert masked.is_code(0)   # 'i' in 'int'
        assert masked.is_code(10)  # '\n'
        assert not masked.is_code(12)  # inside '// this...'

    def test_block_comment(self):
        src = "int x = 1; /* block comment */ int y = 2;"
        masked = mask_comments_and_strings(src)
        assert masked.is_code(0)
        assert not masked.is_code(12)  # inside block comment
        assert masked.is_code(32)  # 'i' in 'int y'

    def test_string_literal(self):
        src = 'printf("hello world"); int x = 1;'
        masked = mask_comments_and_strings(src)
        assert masked.is_code(0)   # 'p' in printf
        assert not masked.is_code(8)   # inside string
        assert masked.is_code(22)  # ';'

    def test_escaped_quote_in_string(self):
        src = r'printf("hello \"world\""); int x = 1;'
        masked = mask_comments_and_strings(src)
        assert not masked.is_code(8)   # inside string
        assert masked.is_code(26)  # ';'

    def test_char_literal(self):
        src = "char c = '\\n'; int x = 1;"
        masked = mask_comments_and_strings(src)
        assert masked.is_code(0)
        assert masked.is_code(17)  # 'i' in 'int'

    def test_for_in_comment_not_detected(self):
        src = "// for (int i = 0; i < N; i++)\nint x = 1;"
        masked = mask_comments_and_strings(src)
        from cuda_opt_agent.transforms.text.lexical import MaskedSource
        import re
        matches = masked.filter_matches(re.compile(r"for\s*\("))
        assert len(matches) == 0

    def test_shared_in_string_not_detected(self):
        src = 'printf("__shared__ memory"); __shared__ float s[256];'
        masked = mask_comments_and_strings(src)
        import re
        matches = masked.filter_matches(re.compile(r"__shared__"))
        assert len(matches) == 1  # Only the real one

    def test_syncthreads_in_comment_not_detected(self):
        src = "/* __syncthreads() */\n__syncthreads();"
        masked = mask_comments_and_strings(src)
        import re
        matches = masked.filter_matches(re.compile(r"__syncthreads"))
        assert len(matches) == 1

    def test_code_spans_contain(self):
        src = '// for loop\nfor (int i = 0; i < N; i++) { }'
        masked = mask_comments_and_strings(src)
        assert code_spans_contain(src, masked.code_ranges, "for (")
        assert not code_spans_contain(src, masked.code_ranges, "for loop")

    def test_nested_block_comments(self):
        """Block comments don't nest in C."""
        src = "/* outer /* inner */ still_comment? */ int x = 1;"
        masked = mask_comments_and_strings(src)
        # After first */, we're back in code
        assert masked.is_code(40)

    def test_multiline_string_preserved(self):
        src = 'const char *s = "line1\\nline2";\nint x = 1;'
        masked = mask_comments_and_strings(src)
        assert masked.is_code(0)
        assert not masked.is_code(16)  # inside string
        assert masked.is_code(30)  # 'i' in int

    def test_transform_preserves_comments(self):
        src = "/* comment */ int x = 1; // line comment"
        masked = mask_comments_and_strings(src)

        def uppercase_code(m: str) -> str:
            return m.upper()

        result = masked.transform(uppercase_code)
        assert "/* comment */" in result
        assert "// line comment" in result
        assert "INT X = 1;" in result