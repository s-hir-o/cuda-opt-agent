"""Timing utilities for kernel execution measurement.

Two strategies:
- HostTimer: host-side time.monotonic() around kernel launch.
  Simple, always available, but includes process startup overhead.
  Accurate enough for kernels > 1ms.
- CudaEventTimer: runs a CUDA-event-instrumented binary that prints
  GPU-only elapsed time to stdout. Requires pre-instrumented binary.
"""

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

from cuda_opt_agent.core.exceptions import BenchmarkError


@dataclass
class TimingMeasurement:
    """Result of a single timing measurement."""

    elapsed_ms: float
    strategy: str  # "host" or "cuda_event"


class KernelTimer(ABC):
    """Abstract timer for measuring kernel execution time."""

    @abstractmethod
    async def measure(
        self, binary_path: str, kernel_name: str | None = None
    ) -> TimingMeasurement:
        """Execute the binary and measure kernel time."""


class HostTimer(KernelTimer):
    """Measure kernel time using host-side time.monotonic().

    Executes the binary as a subprocess and measures wall-clock time.
    Includes process startup, kernel launch, and synchronization overhead.
    Accurate enough for kernels > 1ms.
    """

    async def measure(
        self, binary_path: str, kernel_name: str | None = None
    ) -> TimingMeasurement:
        t0 = time.monotonic()
        try:
            proc = await asyncio.create_subprocess_exec(
                binary_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await asyncio.wait_for(proc.communicate(), timeout=120)
        except FileNotFoundError as exc:
            raise BenchmarkError(f"Binary not found: {binary_path}") from exc
        except asyncio.TimeoutError as exc:
            raise BenchmarkError(f"Benchmark timed out: {binary_path}") from exc

        if proc.returncode != 0:
            stderr = proc.stderr.decode(errors="replace")[:500]
            raise BenchmarkError(
                f"Benchmark exited with code {proc.returncode}: {stderr}"
            )

        elapsed_s = time.monotonic() - t0
        return TimingMeasurement(
            elapsed_ms=elapsed_s * 1000.0,
            strategy="host",
        )


class CudaEventTimer(KernelTimer):
    """Measure GPU-only kernel time by parsing CUDA event output.

    The binary is expected to instrument its own kernel with
    cudaEventRecord/elapsed and print the elapsed milliseconds
    to stdout as a single float value (e.g., "1.234\\n").

    No wrapper source is generated — the binary must be pre-instrumented
    (e.g., by a profiling harness or custom main()).
    """

    async def measure(
        self, binary_path: str, kernel_name: str | None = None
    ) -> TimingMeasurement:
        """Run the CUDA-event-instrumented binary and parse elapsed time."""
        try:
            proc = await asyncio.create_subprocess_exec(
                binary_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=120
            )
        except FileNotFoundError as exc:
            raise BenchmarkError(f"Binary not found: {binary_path}") from exc
        except asyncio.TimeoutError as exc:
            raise BenchmarkError(f"Benchmark timed out: {binary_path}") from exc

        if proc.returncode != 0:
            err = stderr.decode(errors="replace")[:500]
            raise BenchmarkError(
                f"Benchmark exited with code {proc.returncode}: {err}"
            )

        output = stdout.decode(errors="replace").strip()
        try:
            elapsed_ms = float(output.splitlines()[-1])
        except (ValueError, IndexError) as exc:
            raise BenchmarkError(
                f"Cannot parse CUDA event time from output: '{output[:200]}'"
            ) from exc

        return TimingMeasurement(
            elapsed_ms=elapsed_ms,
            strategy="cuda_event",
        )
