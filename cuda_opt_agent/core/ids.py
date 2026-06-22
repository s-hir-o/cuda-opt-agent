"""Deterministic ID generators for runs, variants, and artifacts."""

from __future__ import annotations

import hashlib
import time
import uuid


def _short_hash(data: str, length: int = 8) -> str:
    return hashlib.sha256(data.encode()).hexdigest()[:length]


def generate_run_id(prefix: str = "run") -> str:
    """Unique run identifier: run_<timestamp>_<hash>."""
    ts = str(time.time_ns())
    return f"{prefix}_{ts}_{_short_hash(ts)}"


def generate_variant_id(kernel_name: str, iteration: int, suffix: str = "") -> str:
    """Deterministic variant ID: <kernel>_i<iter>[_<suffix>]."""
    base = f"{kernel_name}_i{iteration}"
    if suffix:
        base = f"{base}_{suffix}"
    return base


def generate_artifact_id(content: str, kind: str = "source") -> str:
    """Content-addressable artifact ID based on SHA-256."""
    h = hashlib.sha256(content.encode()).hexdigest()[:16]
    return f"{kind}_{h}"


def generate_uuid() -> str:
    """Standard UUID4 for when uniqueness matters more than readability."""
    return str(uuid.uuid4())
