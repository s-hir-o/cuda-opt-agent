"""Structural validation for transformed CUDA kernel source.

Checks syntactic plausibility and detects invented placeholder symbols.
Optionally runs ``nvcc --cuda`` when *strict* mode is enabled.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from typing import Any

from cuda_opt_agent.schemas.transform import TransformResult
from cuda_opt_agent.transforms.text.lexical import (
    MaskedSource,
    mask_comments_and_strings,
)


class ValidationError(Exception):
    """Raised when a transformed kernel fails validation."""


def validate_transform(
    source: str,
    original_source: str | None = None,
    result: TransformResult | None = None,
    strict: bool = False,
    nvcc_path: str | None = None,
    arch: str = "sm_70",
    check_comments: bool = True,
) -> bool:
    """Validate transformed kernel source for syntactic safety.

    Parameters
    ----------
    source:
        The transformed CUDA source to validate.
    original_source:
        The original source (used to verify comments/strings preserved).
    result:
        Optional TransformResult to attach errors to.
    strict:
        If True, attempt an nvcc compilation check.
    nvcc_path:
        Path to the nvcc binary. Auto-detected if None.
    arch:
        Target GPU architecture for strict compilation.
    check_comments:
        If True (default), verify comment/string preservation.  Transforms
        that intentionally generate entirely new source (e.g. kernel fusion)
        may set this to False.

    Returns
    -------
    bool
        True if validation passes, False otherwise.
    """
    errors: list[str] = []

    masked = mask_comments_and_strings(source)

    # 1. Balanced braces (masked)
    depth = 0
    for i, ch in enumerate(source):
        if not masked.is_code(i):
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        if depth < 0:
            errors.append("Unmatched closing brace '}'")
            break
    if depth > 0:
        errors.append(f"Unclosed braces: {depth} extra '{{'")

    # 2. Balanced parentheses (masked)
    p_depth = 0
    for i, ch in enumerate(source):
        if not masked.is_code(i):
            continue
        if ch == "(":
            p_depth += 1
        elif ch == ")":
            p_depth -= 1
        if p_depth < 0:
            errors.append("Unmatched closing parenthesis ')'")
            break
    if p_depth > 0:
        errors.append(f"Unclosed parentheses: {p_depth} extra '('")

    # 3. Balanced brackets (code regions only)
    b_depth = 0
    for i, ch in enumerate(source):
        if not masked.is_code(i):
            continue
        if ch == "[":
            b_depth += 1
        elif ch == "]":
            b_depth -= 1
            if b_depth < 0:
                errors.append("Unmatched closing bracket ']'")
                break
    if b_depth > 0:
        errors.append(f"Unclosed brackets: {b_depth} extra '['")

    # 4. Unclosed string literals (must scan ALL characters)
    in_string = False
    escape_next = False
    for ch in source:
        if escape_next:
            escape_next = False
            continue
        if ch == "\\":
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
    if in_string:
        errors.append("Unclosed string literal")

    # 5. Unclosed block comments
    if "/*" in source:
        last_open = source.rfind("/*")
        last_close = source.rfind("*/")
        if last_close < last_open:
            errors.append("Unclosed block comment")

    # 6. Valid kernel declaration
    kernel_pattern = re.compile(r"__global__\s+void\s+\w+\s*\(")
    kernels = kernel_pattern.findall(source)
    if not kernels:
        errors.append("No __global__ void kernel declaration found")

    # 7. Non-empty kernel body (only fail if ALL kernels are empty)
    kernel_bodies_empty = []
    for km in kernel_pattern.finditer(source):
        brace_pos = source.find("{", km.end())
        if brace_pos < 0:
            errors.append(f"Kernel '{km.group()}' has no opening brace")
            continue
        d = 0
        body_end = -1
        for i in range(brace_pos, len(source)):
            if source[i] == "{":
                d += 1
            elif source[i] == "}":
                d -= 1
                if d == 0:
                    body_end = i
                    break
        if body_end < 0:
            continue
        body = source[brace_pos + 1 : body_end].strip()
        kernel_bodies_empty.append(not body)
    if kernel_bodies_empty and all(kernel_bodies_empty):
        errors.append("Kernel has empty body")

    # 8. Invented placeholder symbols
    placeholder_patterns = [
        r"\b\w+_base\b",
        r"\bplaceholder_\w+\b",
        r"\bTMP_VAR_\w+\b",
    ]
    for pp in placeholder_patterns:
        matches = [m.group() for m in masked.filter_matches(re.compile(pp))]
        if matches:
            errors.append(f"Invented placeholder symbols: {matches}")

    # 9. Comment/string preservation check (optional)
    if check_comments and original_source is not None:
        if not _comments_strings_preserved(original_source, source):
            errors.append(
                "Transform modified comment or string literal content"
            )

    # 10. Strict nvcc compilation check
    if strict and not errors:
        compiler = nvcc_path or shutil.which("nvcc")
        if compiler is None:
            pass  # nvcc not available; skip strict check
        else:
            ok, msg = _nvcc_compile_check(source, compiler, arch)
            if not ok:
                errors.append(f"nvcc compilation failed: {msg}")

    if errors:
        if result is not None:
            result.success = False
            result.error_message = "; ".join(errors)
        return False

    if result is not None:
        result.success = True
        result.error_message = ""
    return True


def _comments_strings_preserved(original: str, transformed: str) -> bool:
    """Verify that comment and string literal content is identical."""
    orig_masked = mask_comments_and_strings(original)
    trans_masked = mask_comments_and_strings(transformed)

    def _extract_non_code(s: str, ms: MaskedSource) -> list[str]:
        spans: list[str] = []
        prev_end = 0
        for start, end in ms.code_ranges:
            if start > prev_end:
                spans.append(s[prev_end:start])
            prev_end = end
        if prev_end < len(s):
            spans.append(s[prev_end:])
        return spans

    return _extract_non_code(original, orig_masked) == _extract_non_code(
        transformed, trans_masked
    )


def _nvcc_compile_check(
    source: str, compiler: str, arch: str
) -> tuple[bool, str]:
    """Attempt to compile *source* with nvcc. Returns (success, message)."""
    tmpdir = tempfile.mkdtemp(prefix="cuda_opt_val_")
    path = os.path.join(tmpdir, "kernel.cu")
    try:
        with open(path, "w") as f:
            f.write(source)
        proc = subprocess.run(
            [compiler, "--cuda", "-arch", arch, "-c", path, "-o", os.devnull],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if proc.returncode == 0:
            return True, ""
        return False, proc.stderr.strip()[:500]
    except Exception as exc:
        return False, str(exc)
    finally:
        try:
            os.remove(path)
            os.rmdir(tmpdir)
        except OSError:
            pass
