"""Tests for lexical masking."""

from cuda_opt_agent.transforms.text.lexical import (
    mask_comments_and_strings,
    code_spans_contain,
)
import pytest
import re


class TestMasking:

    def test_single_line_comment(self):
        src = "int x = 1; // comment\nint y = 2;"
        masked = mask_comments_and_strings(src)
        assert masked.is_code(src.index("//")) is False
        assert masked.is_code(src.index("int y")) is True

    def test_block_comment(self):
        src = "int x = 1; /* block\ncomment */ int y = 2;"
        masked = mask_comments_and_strings(src)
        assert masked.is_code(src.index("block")) is False
        assert masked.is_code(src.index("int y")) is True

    def test_string_literal(self):
        src = 'printf("hello { world"); int x = 1;'
        masked = mask_comments_and_strings(src)
        assert masked.is_code(src.index("hello")) is False
        assert masked.is_code(src.index("int x")) is True

    def test_char_literal(self):
        src = "char c = '{'; int x = 1;"
        masked = mask_comments_and_strings(src)
        assert masked.is_code(src.index("'{'")) is False

    def test_braces_in_comment_not_code(self):
        src = "/* { */ int x = 1; /* } */"
        masked = mask_comments_and_strings(src)
        assert masked.is_code(src.index("{")) is False
        assert masked.is_code(src.index("}")) is False

    def test_filter_skips_comments(self):
        src = "/* for */\nfor (int i = 0; i < 10; i++) {}"
        masked = mask_comments_and_strings(src)
        matches = masked.filter_matches(re.compile(r"for\s*\("))
        assert len(matches) == 1
        assert matches[0].start() > src.index("*/")

    def test_code_spans_contain_true(self):
        src = "/* __shared__ */ __shared__ float s[32];"
        masked = mask_comments_and_strings(src)
        assert code_spans_contain(src, masked.code_ranges, "__shared__") is True

    def test_code_spans_contain_false(self):
        src = "/* __shared__ */ float s[32];"
        masked = mask_comments_and_strings(src)
        assert code_spans_contain(src, masked.code_ranges, "__shared__") is False

    def test_transform_preserves_length(self):
        src = "int x = 1; // comment"
        masked = mask_comments_and_strings(src)
        result = masked.transform(lambda s: s)
        assert len(result) == len(src)
        assert result == src

    def test_transform_rejects_length_change(self):
        src = "int x = 1; // comment"
        masked = mask_comments_and_strings(src)
        with pytest.raises(ValueError, match="equal length"):
            masked.transform(lambda s: s + "EXTRA")

    def test_escaped_quotes(self):
        src = r'printf("hello \" world"); int x = 1;'
        masked = mask_comments_and_strings(src)
        assert masked.is_code(src.index("int x")) is True