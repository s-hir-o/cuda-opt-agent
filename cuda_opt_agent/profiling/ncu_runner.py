"""Subprocess wrapper for NVIDIA Nsight Compute (ncu).

Runs ncu with --page raw --csv and captures stdout for parsing.
"""

from __future__ import annotations

import asyncio
import shutil
from pathlib import Path

from cuda_opt_agent.core.config import ProfilingConfig
from cuda_opt_agent.core.exceptions import ProfilerUnavailableError, ProfilingError
from cuda_opt_agent.build.nvcc_runner import NVCCRunner
from cuda_opt_agent.profiling.ncu_parser import parse_ncu_csv
from cuda_opt_agent.profiling.profiler import Profiler
from cuda_opt_agent.schemas.kernel import KernelSource, KernelVariant
from cuda_opt_agent.schemas.metrics import ProfileMetrics


class NCURunner:
    """Execute ncu as a subprocess and return raw CSV output."""

    def __init__(self, config: ProfilingConfig) -> None:
        self._ncu_path = config.ncu_path
        self._metrics = config.metrics

    def _command(self, *args: str) -> list[str]:
        executable = shutil.which(self._ncu_path) or self._ncu_path
        if Path(executable).suffix.lower() in {".bat", ".cmd"}:
            # The Windows ncu launcher is a batch file. Python's subprocess
            # cannot execute it directly, so use the executable it launches.
            native = (
                Path(executable).parent / "target" / "windows-desktop-win7-x64" / "ncu.exe"
            )
            if native.is_file():
                executable = str(native)
        return [executable, *args]

    async def is_available(self) -> bool:
        try:
            proc = await asyncio.create_subprocess_exec(
                *self._command("--version"),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await proc.wait()
            return proc.returncode == 0
        except FileNotFoundError:
            return False

    async def run_ncu(
        self,
        binary_path: str,
        kernel_name: str | None = None,
        arch: str = "sm_80",
    ) -> str:
        """Run ncu on a compiled binary. Returns raw CSV stdout.

        Args:
            binary_path: Path to the compiled CUDA executable.
            kernel_name: Optional kernel name filter.
            arch: Target architecture (used for kernel matching).

        Returns:
            Raw ncu CSV output string.

        Raises:
            ProfilerUnavailableError: If ncu binary not found.
            ProfilingError: If ncu returns non-zero exit code.
        """
        cmd = [
            "--set", "full",
            "--target-processes", "all",
            "--page", "raw",
            "--csv",
            "--launch-count", "1",
        ]

        if self._metrics:
            cmd.extend(["--metrics", ",".join(self._metrics)])

        if kernel_name:
            cmd.extend(["--kernel-name", kernel_name])

        cmd.append(binary_path)

        try:
            proc = await asyncio.create_subprocess_exec(
                *self._command(*cmd),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=300
            )
        except FileNotFoundError as exc:
            raise ProfilerUnavailableError(
                f"ncu not found at '{self._ncu_path}'"
            ) from exc
        except asyncio.TimeoutError as exc:
            raise ProfilingError("ncu timed out after 300 seconds") from exc

        if proc.returncode != 0:
            err = (stderr or stdout).decode(errors="replace")[:2000]
            raise ProfilingError(
                f"ncu exited with code {proc.returncode}: {err}",
            )

        return stdout.decode(errors="replace")


class NCUProfiler(Profiler):
    """Compile a kernel to a host executable, run ncu, and parse metrics."""

    def __init__(self, ncu_runner: NCURunner, nvcc_runner: NVCCRunner) -> None:
        self._ncu = ncu_runner
        self._nvcc = nvcc_runner

    async def is_available(self) -> bool:
        return await self._ncu.is_available() and await self._nvcc.is_available()

    async def profile(
        self,
        kernel: KernelSource | KernelVariant,
        run_id: str,
        arch: str,
    ) -> ProfileMetrics:
        compilation = await self._nvcc.compile_executable(kernel, arch=arch)
        if not compilation.success or not compilation.binary_path:
            diagnostics = "; ".join(d.message for d in compilation.diagnostics[:3])
            details = diagnostics or compilation.stderr[:500] or "unknown nvcc error"
            raise ProfilingError(f"Cannot profile kernel; executable compile failed: {details}")

        kernel_name = _kernel_entry_point(kernel)
        kernel_filter = kernel_name if kernel_name != "main" else None
        raw_csv = await self._ncu.run_ncu(
            binary_path=compilation.binary_path,
            kernel_name=kernel_filter,
            arch=arch,
        )
        metrics = parse_ncu_csv(raw_csv, kernel_name=kernel_name)
        if not metrics.arch:
            metrics.arch = arch
        if metrics.metadata is None:
            metrics.metadata = {}
        metrics.metadata["run_id"] = run_id
        metrics.metadata["binary_path"] = compilation.binary_path
        return metrics


def _kernel_entry_point(kernel: KernelSource | KernelVariant) -> str:
    entry_point = getattr(kernel, "entry_point", None)
    if entry_point:
        return entry_point
    if isinstance(kernel, KernelSource):
        return kernel.short_name
    return kernel.variant_id
