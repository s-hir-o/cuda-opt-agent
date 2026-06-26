
"""Detect memory access patterns: stride, coalescing, sequential."""

from __future__ import annotations

import re
from collections import defaultdict

from cuda_opt_agent.schemas.analysis import MemAccessPattern, PTXInstruction

_STRIDE_RE = re.compile(r"$$(%\w+)\+(\d+)$$")


def detect_access_patterns(instructions: list[PTXInstruction]) -> list[MemAccessPattern]:
    """Analyze global memory instructions for access patterns.

    Groups loads/stores by base register and detects stride patterns.
    """
    patterns: list[MemAccessPattern] = []

    # Group memory instructions by base address register
    groups: dict[str, list[PTXInstruction]] = defaultdict(list)
    for instr in instructions:
        if instr.memory_space != "global":
            continue
        base = _extract_base_register(instr)
        if base:
            groups[base].append(instr)

    for base_reg, group in groups.items():
        if len(group) < 2:
            continue

        offsets = []
        for instr in group:
            off = instr.memory_offset
            if off is not None:
                offsets.append(off)

        if len(offsets) < 2:
            # No constant offsets found — likely non-strided
            patterns.append(MemAccessPattern(
                pattern_type="unknown",
                is_coalesced=False,
                affected_instructions=[i.index for i in group],
                description=f"Inconsistent offsets for base {base_reg}",
            ))
            continue

          # Compute stride
        strides = [offsets[i + 1] - offsets[i] for i in range(len(offsets) - 1)]
        unique_strides = set(strides)

        if all(s == 0 for s in strides):
            patterns.append(MemAccessPattern(
                pattern_type="broadcast",
                stride_bytes=0,
                is_coalesced=True,
                affected_instructions=[i.index for i in group],
                description=f"Broadcast access on {base_reg}",
            ))
        elif len(unique_strides) == 1:
            stride = unique_strides.pop()
            is_coalesced = stride in (4, 8, 16)  # 4B, 8B, 16B vector widths
            patterns.append(MemAccessPattern(
                pattern_type="stride",
                stride_bytes=stride,
                is_coalesced=is_coalesced,
                affected_instructions=[i.index for i in group],
                description=f"Stride-{stride}B access on {base_reg}",
            ))
        else:
            patterns.append(MemAccessPattern(
                pattern_type="random",
                is_coalesced=False,
                affected_instructions=[i.index for i in group],
                description=f"Random/irregular access on {base_reg}",
            ))

    return patterns


def _extract_base_register(instr: PTXInstruction) -> str | None:
    """Extract the base register from a memory operand."""
    for op in instr.operands:
        m = re.search(r"\[(%\w+)", op)
        if m:
            return m.group(1)
    return None
