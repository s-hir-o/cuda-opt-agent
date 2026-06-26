
"""Parse PTX source text into a list of PTXInstruction objects.

Handles: labels, directives, comments, predicates, typed opcodes,
register operands, memory operands, immediates.
"""

from __future__ import annotations

import re
from typing import Any

from cuda_opt_agent.analysis.ptx.instruction_analyzer import classify_opcode
from cuda_opt_agent.core.exceptions import PTXParseError
from cuda_opt_agent.schemas.analysis import OpcodeClass, PTXInstruction

# Register references: %r1, %f0, %rd3, %p1, %rd10, etc.
_REG_RE = re.compile(r"%[a-z]+\d+")

# Special (hardware) registers — excluded from dependency tracking
_SPECIAL_REG_PREFIXES = {"tid", "ntid", "ctaid", "nctaid", "laneid", "warpid", "smid"}

# Memory address pattern: [%reg] or [%reg+expr]
_MEM_ADDR_RE = re.compile(r"\[([^\]]+)\]")

def parse_ptx(ptx_source: str) -> list[PTXInstruction]:
    """Parse PTX source into a list of PTXInstruction objects.

    Args:
        ptx_source: Complete PTX source text.

    Returns:
        Ordered list of parsed instructions (directives and labels skipped).

    Raises:
        PTXParseError: If parsing encounters an unrecoverable error.
    """
    # Strip block comments
    source = re.sub(r"/\*.*?\*/", "", ptx_source, flags=re.DOTALL)

    instructions: list[PTXInstruction] = []
    idx = 0

    for line_num, raw_line in enumerate(source.splitlines(), 1):
        # Strip line comments and whitespace
        line = raw_line.split("//")[0].strip()
        if not line:
            continue

        # Skip directives (start with .)
        if line.startswith("."):
            continue

        # Skip labels (word followed by colon, nothing else meaningful)
        if re.match(r"^[\w_]+:\s*$", line):
            continue

        # If line has a label prefix and then an instruction, strip the label
        label_match = re.match(r"^[\w_]+:\s*(.+)", line)
        if label_match:
            line = label_match.group(1).strip()
            if not line or line.startswith("."):
                continue

        instr = _parse_instruction_line(line, line_num)
        if instr is not None:
            instr.index = idx
            idx += 1
            instructions.append(instr)

    return instructions


def _parse_instruction_line(line: str, line_num: int) -> PTXInstruction | None:
    """Parse a single PTX instruction line."""

    # Match: [predicate] opcode [operands];
    m = re.match(
        r"^(?:\s*(@!?\s*%\w+)\s+)?"  # optional predicate
        r"([\w.]+)"                     # opcode (e.g., add.s32, ld.global.f32)
        r"(?:\s+(.*?))?"                # operands
        r"\s*;\s*$",                    # semicolon terminator
        line,
    )
    if not m:
        return None

    predicate_raw = m.group(1)
    opcode = m.group(2)
    operands_str = (m.group(3) or "").strip()

    predicate = predicate_raw.replace(" ", "") if predicate_raw else None
    operands = _split_operands(operands_str)
    opcode_class = classify_opcode(opcode)
    memory_space = _extract_memory_space(opcode)
    is_branch = opcode.split(".")[0] in ("bra", "brx", "call", "ret", "exit")
    memory_offset = _extract_memory_offset(operands)

    dest_regs, src_regs = _classify_registers(opcode, operands)

    return PTXInstruction(
        index=0,  # set by caller
        raw=line,
        opcode=opcode,
        opcode_class=opcode_class,
        operands=operands,
        predicate=predicate,
        dest_registers=dest_regs,
        src_registers=src_regs,
        memory_space=memory_space,
        memory_offset=memory_offset,
        is_branch=is_branch,
        line_number=line_num,
    )


def _split_operands(text: str) -> list[str]:
    """Split comma-separated operands respecting bracket nesting."""
    if not text:
        return []
    operands: list[str] = []
    current: list[str] = []
    depth = 0
    for ch in text:
        if ch == "[":
            depth += 1
            current.append(ch)
        elif ch == "]":
            depth -= 1
            current.append(ch)
        elif ch == "," and depth == 0:
            operands.append("".join(current).strip())
            current = []
        else:
            current.append(ch)
    tail = "".join(current).strip()
    if tail:
        operands.append(tail)
    return operands


def _extract_memory_space(opcode: str) -> str | None:
    """Extract memory space qualifier from opcode (e.g., 'global' from ld.global.f32)."""
    parts = opcode.split(".")
    spaces = {"global", "shared", "local", "const", "param", "tex"}
    for p in parts[1:]:
        if p in spaces:
            return p
    return None


def _extract_memory_offset(operands: list[str]) -> int | None:
    """Extract constant offset from a memory operand like [%rd1+16]."""
    for op in operands:
        m = _MEM_ADDR_RE.search(op)
        if m:
            addr = m.group(1)
            offset_match = re.search(r"\+(\d+)", addr)
            if offset_match:
                return int(offset_match.group(1))
    return None


def _classify_registers(
    opcode: str, operands: list[str]
) -> tuple[list[str], list[str]]:
    """Classify registers as destination (written) or source (read).

    For store instructions: all registers are sources.
    For other instructions: first operand's register is destination, rest are sources.
    Special hardware registers (tid, ctaid, etc.) are excluded.
    """
    base = opcode.split(".")[0]

    def _filter_special(regs: list[str]) -> list[str]:
        return [r for r in regs if not _is_special(r)]

    def _is_special(reg: str) -> bool:
        name = reg.lstrip("%").split(".")[0]
        return name in _SPECIAL_REG_PREFIXES

    if base in ("st",):
        all_regs: list[str] = []
        for op in operands:
            all_regs.extend(_REG_RE.findall(op))
        return [], _filter_special(all_regs)

    if base in ("bra", "brx", "ret", "exit", "bar", "sync"):
        return [], []

    dest: list[str] = []
    if operands:
        first_regs = _REG_RE.findall(operands[0])
        if first_regs and not _is_special(first_regs[0]):
            dest = [first_regs[0]]

    src: list[str] = []
    for op in operands[1:]:
        for r in _REG_RE.findall(op):
            if not _is_special(r):
                src.append(r)

    return dest, src
