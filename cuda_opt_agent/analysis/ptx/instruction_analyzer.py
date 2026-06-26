"""Classify PTX opcodes into high-level instruction categories."""

from __future__ import annotations

from cuda_opt_agent.schemas.analysis import OpcodeClass

# Opcode base → classification mapping
_OPCODE_MAP: dict[str, OpcodeClass] = {
    # Arithmetic / logic
    "add": OpcodeClass.ALU, "sub": OpcodeClass.ALU, "mul": OpcodeClass.ALU,
    "div": OpcodeClass.ALU, "rem": OpcodeClass.ALU, "neg": OpcodeClass.ALU,
    "abs": OpcodeClass.ALU, "min": OpcodeClass.ALU, "max": OpcodeClass.ALU,
    "and": OpcodeClass.ALU, "or": OpcodeClass.ALU, "xor": OpcodeClass.ALU,
    "not": OpcodeClass.ALU, "shl": OpcodeClass.ALU, "shr": OpcodeClass.ALU,
    "bfind": OpcodeClass.ALU, "clz": OpcodeClass.ALU, "popc": OpcodeClass.ALU,
    "lop3": OpcodeClass.ALU, "prmt": OpcodeClass.ALU, "sad": OpcodeClass.ALU,
    "brev": OpcodeClass.ALU, "bfe": OpcodeClass.ALU, "bfi": OpcodeClass.ALU,
    # Fused multiply-add
    "fma": OpcodeClass.FMA, "mad": OpcodeClass.FMA,
    # Global memory
    "ld": OpcodeClass.MEMORY_LOAD, "ldu": OpcodeClass.MEMORY_LOAD,
    "ldg": OpcodeClass.MEMORY_LOAD, "lds": OpcodeClass.MEMORY_LOAD,
    "st": OpcodeClass.MEMORY_STORE, "stg": OpcodeClass.MEMORY_STORE,
    "sts": OpcodeClass.MEMORY_STORE,
    # Shared memory (handled via memory_space check below)
    # Control flow
    "bra": OpcodeClass.CONTROL_FLOW, "brx": OpcodeClass.CONTROL_FLOW,
    "call": OpcodeClass.CONTROL_FLOW, "ret": OpcodeClass.CONTROL_FLOW,
    "exit": OpcodeClass.CONTROL_FLOW,
    "brkpt": OpcodeClass.CONTROL_FLOW, "nanosleep": OpcodeClass.CONTROL_FLOW,
    # Comparison / predicate
    "setp": OpcodeClass.COMPARISON, "set": OpcodeClass.COMPARISON,
    "selp": OpcodeClass.COMPARISON, "slct": OpcodeClass.COMPARISON,
    # Conversion
    "cvt": OpcodeClass.CONVERSION, "cvt.pack": OpcodeClass.CONVERSION,
    # Atomics
    "atom": OpcodeClass.ATOMIC, "red": OpcodeClass.ATOMIC,
    "cas": OpcodeClass.ATOMIC,
    # Sync
    "bar": OpcodeClass.SYNC, "membar": OpcodeClass.SYNC,
    "atombar": OpcodeClass.SYNC,
    # Movement
    "mov": OpcodeClass.ALU, "movmatrix": OpcodeClass.ALU,
    "prmt": OpcodeClass.ALU,
}


def classify_opcode(opcode: str) -> OpcodeClass:
    """Classify a PTX opcode string into an OpcodeClass.

    Uses the base opcode (before the first dot) for lookup, then
    refines based on memory space qualifiers.
    """
    parts = opcode.lower().split(".")
    base = parts[0]

    cls = _OPCODE_MAP.get(base, OpcodeClass.OTHER)

    # Refine shared memory variants
    if cls in (OpcodeClass.MEMORY_LOAD, OpcodeClass.MEMORY_STORE):
        if "shared" in parts or "lds" in base or "sts" in base:
            if cls == OpcodeClass.MEMORY_LOAD:
                return OpcodeClass.SHARED_LOAD
            return OpcodeClass.SHARED_STORE

    return cls
