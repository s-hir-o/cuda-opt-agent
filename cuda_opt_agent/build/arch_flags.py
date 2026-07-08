"""Architecture flag utilities for nvcc.

Maps architecture strings (sm_80, sm_89, sm_90) to nvcc flags.
Provides validation and common flag sets.
"""

from __future__ import annotations

import re

from cuda_opt_agent.core.exceptions import ConfigurationError

# Known architectures and their GPU names
_KNOWN_ARCHS: dict[str, str] = {
    "sm_70": "Volta (V100)",
    "sm_75": "Turing (T4, RTX 2080)",
    "sm_80": "Ampere (A100)",
    "sm_86": "Ampere (RTX 3090)",
    "sm_89": "Ada Lovelace (RTX 4090, L40)",
    "sm_90": "Hopper (H100)",
}

_ARCH_RE = re.compile(r"^sm_(\d+)$")


def validate_arch(arch: str) -> str:
    """Validate and normalize an architecture string.

    Args:
        arch: Architecture string like 'sm_80', 'sm_89', 'sm_90'.

    Returns:
        The normalized architecture string.

    Raises:
        ConfigurationError: If the architecture string is malformed.
    """
    arch = arch.strip().lower()
    if not _ARCH_RE.match(arch):
        raise ConfigurationError(
            f"Invalid architecture '{arch}'. Expected format: sm_XX (e.g., sm_80)."
        )
    return arch


def is_known_arch(arch: str) -> bool:
    """Return True if arch is in the known architecture list."""
    return arch.lower() in _KNOWN_ARCHS


def gpu_name(arch: str) -> str:
    """Return the human-readable GPU name for an architecture, or 'Unknown'."""
    return _KNOWN_ARCHS.get(arch.lower(), "Unknown GPU")


def list_archs() -> list[str]:
    """Return all known architecture strings."""
    return sorted(_KNOWN_ARCHS.keys())


def ptx_arch_flag(arch: str) -> list[str]:
    """Return nvcc flags for PTX-only compilation at the given architecture."""
    arch = validate_arch(arch)
    return ["-ptx", f"-arch={arch}"]


def cubin_arch_flag(arch: str) -> list[str]:
    """Return nvcc flags for cubin compilation at the given architecture."""
    arch = validate_arch(arch)
    return ["-cubin", f"-arch={arch}"]


def executable_arch_flag(arch: str) -> list[str]:
    """Return nvcc flags for executable compilation (for profiling)."""
    arch = validate_arch(arch)
    return [f"-arch={arch}"]


def common_flags(
    optimization_level: int = 3,
    fast_math: bool = False,
    lineinfo: bool = True,
    extra: list[str] | None = None,
) -> list[str]:
    """Build a common set of nvcc flags.

    Args:
        optimization_level: -O level (0-3). Default 3.
        fast_math: Enable --use_fast_math.
        lineinfo: Enable -lineinfo for profiler source mapping.
        extra: Additional flags to append.

    Returns:
        List of nvcc flag strings.

    Raises:
        ConfigurationError: If optimization_level is not in {0, 1, 2, 3}.
    """
    if optimization_level not in (0, 1, 2, 3):
        raise ConfigurationError(
            f"Invalid optimization level {optimization_level}. "
            "nvcc supports -O0 through -O3."
        )

    flags = [f"-O{optimization_level}"]

    if fast_math:
        flags.append("--use_fast_math")
    if lineinfo:
        flags.append("-lineinfo")

    if extra:
        flags.extend(extra)

    return flags
