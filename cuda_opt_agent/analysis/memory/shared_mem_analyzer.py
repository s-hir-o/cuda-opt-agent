"""Detect bank conflicts in shared memory accesses."""

from __future__ import annotations

import re

from cuda_opt_agent.schemas.analysis import MemAccessPattern, PTXInstruction

# PTX has 32 banks, 4 bytes each for 32-bit accesses
_NUM_BANKS = 32
_BANK_WIDTH_BYTES = 4


def detect_bank_conflicts(instructions: list[PTXInstruction]) -> tuple[int, list[MemAccessPattern]]:
    """Analyze shared memory accesses for potential bank conflicts.

    Returns:
        Tuple of (conflict_count, list of MemAccessPattern for conflicted accesses).
    """
    patterns: list[MemAccessPattern] = []
    conflict_count = 0

    shared_instrs = [
        i for i in instructions
        if i.memory_space == "shared"
    ]

    # Group by line proximity (within 1 instruction = likely same warp)
    # Simplified: check if consecutive shared mem accesses have stride = N * bank_width
    offsets: list[tuple[int, int | None]] = []  # (index, offset)
    for instr in shared_instrs:
        offsets.append((instr.index, instr.memory_offset))

    # Check for patterns that cause bank conflicts
    # A stride of 32 * 4 = 128 bytes maps all accesses to the same bank
    for i in range(len(offsets)):
        for j in range(i + 1, min(i + 32, len(offsets))):
            idx_a, off_a = offsets[i]
            idx_b, off_b = offsets[j]
            if off_a is not None and off_b is not None:
                diff = abs(off_b - off_a)
                bank_a = (off_a // _BANK_WIDTH_BYTES) % _NUM_BANKS
                bank_b = (off_b // _BANK_WIDTH_BYTES) % _NUM_BANKS
                if bank_a == bank_b and diff > 0:
                    conflict_count += 1

    # Detect stride-based conflicts
    for instr in shared_instrs:
        if instr.memory_offset is not None:
            bank = (instr.memory_offset // _BANK_WIDTH_BYTES) % _NUM_BANKS
            # Stride of 128 bytes (32 banks * 4 bytes) → all same bank
            if instr.memory_offset % (_NUM_BANKS * _BANK_WIDTH_BYTES) == 0 and instr.memory_offset > 0:
                patterns.append(MemAccessPattern(
                    pattern_type="bank_conflict",
                    stride_bytes=instr.memory_offset,
                    bank_conflicts=_NUM_BANKS,
                    is_coalesced=False,
                    affected_instructions=[instr.index],
                    description=f"Shared memory stride {instr.memory_offset}B causes {_NUM_BANKS}-way bank conflict",
                ))

    return conflict_count, patterns
