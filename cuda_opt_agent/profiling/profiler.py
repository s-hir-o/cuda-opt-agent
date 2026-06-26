"""Abstract profiler interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

from cuda_opt_agent.schemas.kernel import KernelSource, KernelVariant
from cuda_opt_agent.schemas.metrics import ProfileMetrics


class Profiler(ABC):
    """Profile a compiled CUDA kernel and return structured metrics."""

    @abstractmethod
    async def profile(
        self,
        kernel: KernelSource | KernelVariant,
        run_id: str,
        arch: str,
    ) -> ProfileMetrics:
        """Profile the kernel. Raises ProfilingError on failure."""

    @abstractmethod
    async def is_available(self) -> bool:
        """Return True if the profiler toolchain is functional."""
