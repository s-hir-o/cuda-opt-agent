"""Lexical masking utility for CUDA source.

Pre-processes source code to identify and mask comments and string literals,
ensuring regex-based transforms operate only on real code tokens.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class MaskedSource:
    """Holds original source and a list of (start, end) code-only ranges.

    All positions are character indices into the original source string.
    Ranges are sorted, non-overlapping, and represent regions that are
    NOT inside comments or string/character literals.
    """

    source: str
    code_ranges: list[tuple[int, int]] = field(default_factory=list)

    def is_code(self, pos: int) -> bool:
        """Return True if *pos* is inside a code region (not comment/string)."""
        for start, end in self.code_ranges:
            if start <= pos < end:
                return True
        return False

    def filter_matches(self, pattern: re.Pattern) -> list[re.Match]:
        """Return regex matches whose start position is in a code region."""
        return [m for m in pattern.finditer(self.source) if self.is_code(m.start())]

    def transform(self, transform_fn: Callable[[str], str]) -> str:
        """Apply *transform_fn* to the code-only portions and reassemble.

        Comments and string literals are preserved verbatim.
        *transform_fn* receives a string composed only of code-region characters
        (with masked regions replaced by spaces) and must return a string of
        equal length (replace in-place only, no insertions/deletions).

        FIX: explicitly enforces the equal-length contract.
        """
        masked = _build_masked_string(self.source, self.code_ranges)
        transformed_masked = transform_fn(masked)
        if len(transformed_masked) != len(masked):
            raise ValueError(
                f"transform_fn must return a string of equal length "
                f"(got {len(transformed_masked)}, expected {len(masked)}). "
                f"In-place replacement only — no insertions or deletions."
            )
        return _reassemble(self.source, transformed_masked, self.code_ranges)


def mask_comments_and_strings(source: str) -> MaskedSource:
    """Identify code-only spans in *source*, skipping comments and literals.

    Handles:
    - Single-line comments: ``// ...``
    - Block comments: ``/* ... */``
    - String literals: ``"..."`` (including escaped quotes)
    - Character literals: ``'...'`` (including escaped quotes)

    Returns a MaskedSource whose code_ranges list contains (start, end) tuples
    for every contiguous run of actual code.
    """
    ranges: list[tuple[int, int]] = []
    code_start: int | None = 0
    i = 0
    length = len(source)

    while i < length:
        ch = source[i]

        # --- Single-line comment ---
        if ch == "/" and i + 1 < length and source[i + 1] == "/":
            if code_start is not None and code_start < i:
                ranges.append((code_start, i))
            code_start = None
            i += 2
            while i < length and source[i] != "\n":
                i += 1
            # The newline itself is code (it's syntactically significant)
            if i < length:
                code_start = i
            continue

        # --- Block comment ---
        if ch == "/" and i + 1 < length and source[i + 1] == "*":
            if code_start is not None and code_start < i:
                ranges.append((code_start, i))
            code_start = None
            i += 2
            while i < length:
                if source[i] == "*" and i + 1 < length and source[i + 1] == "/":
                    i += 2
                    code_start = i
                    break
                i += 1
            else:
                # Unterminated block comment — rest of file is comment
                code_start = None
            continue

        # --- String literal ---
        if ch == '"':
            if code_start is not None and code_start < i:
                ranges.append((code_start, i))
            code_start = None
            i += 1
            while i < length:
                if source[i] == "\\" and i + 1 < length:
                    i += 2  # skip escaped char
                elif source[i] == '"':
                    i += 1
                    code_start = i
                    break
                else:
                    i += 1
            else:
                code_start = None
            continue

        # --- Character literal ---
        if ch == "'":
            if code_start is not None and code_start < i:
                ranges.append((code_start, i))
            code_start = None
            i += 1
            while i < length:
                if source[i] == "\\" and i + 1 < length:
                    i += 2
                elif source[i] == "'":
                    i += 1
                    code_start = i
                    break
                else:
                    i += 1
            else:
                code_start = None
            continue

        i += 1

    # Final code region
    if code_start is not None and code_start < length:
        ranges.append((code_start, length))

    return MaskedSource(source=source, code_ranges=ranges)


def code_spans_contain(source: str, ranges: list[tuple[int, int]], text: str) -> bool:
    """Check if *text* appears in the source within any code region."""
    start = 0
    while True:
        idx = source.find(text, start)
        if idx < 0:
            return False
        if any(s <= idx < e for s, e in ranges):
            return True
        start = idx + 1


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_MASK_CHAR = " "  # replaces masked regions; preserves length


def _build_masked_string(source: str, ranges: list[tuple[int, int]]) -> str:
    """Build a string where non-code characters are replaced with spaces."""
    masked = list(source)
    # Mark everything as masked first
    for i in range(len(masked)):
        masked[i] = _MASK_CHAR
    # Un-mask code regions
    for start, end in ranges:
        for i in range(start, end):
            masked[i] = source[i]
    return "".join(masked)


def _reassemble(
    original: str, transformed_masked: str, ranges: list[tuple[int, int]]
) -> str:
    """Combine transformed code regions with original comments/strings."""
    result = list(original)
    for start, end in ranges:
        for i in range(start, end):
            result[i] = transformed_masked[i]
    return "".join(result)