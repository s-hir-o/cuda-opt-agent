"""Subprocess wrapper for NVIDIA Nsight Compute (ncu).

Runs ncu with --page raw --csv and captures stdout for parsing.
"""

from __future__ import annotations

import asyncio
import shlex
from pathlib import Path

from cuda_opt_agent.core.config import ProfilingConfig
from cuda_opt_agent.core.exceptions import ProfilerUnavailableError, ProfilingError


class NCURunner:
    """Execute ncu as a subprocess and return raw CSV output."""

    def __init__(self, config: ProfilingConfig) -> None:
        self._ncu_path = config.ncu_path
        self._metrics = config.metrics

    async def is_available(self) -> bool:
        try:
            proc = await asyncio.create_subprocess_exec(
                self._ncu_path, "--version",
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
            self._ncu_path,
            "--set", "full",
            "--target-processes", "all",
            "--page", "raw",
            "--csv",
            "--launch-count", "1",
        ]

        for metric in self._metrics:
            cmd.extend(["--metrics", metric])

        if kernel_name:
            cmd.extend(["--kernel-name", kernel_name])

        cmd.append(binary_path)

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
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
            err = stderr.decode(errors="replace")[:2000]
            raise ProfilingError(
                f"ncu exited with code {proc.returncode}: {err}",
            )

        return stdout.decode(errors="replace")
