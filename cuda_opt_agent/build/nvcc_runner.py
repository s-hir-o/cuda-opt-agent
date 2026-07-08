"""NVCC compiler wrapper.

Compiles CUDA source to cubin, PTX, or executable.
Parses nvcc diagnostics into CompilationDiagnostic objects.
Returns CompilationResult for every compilation attempt.
"""

from __future__ import annotations

import asyncio
import os
import re
import shutil
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

from cuda_opt_agent.build.arch_flags import (
    common_flags,
    cubin_arch_flag,
    executable_arch_flag,
    ptx_arch_flag,
)
from cuda_opt_agent.core.config import CompilerConfig
from cuda_opt_agent.core.exceptions import PTXExtractionError
from cuda_opt_agent.schemas.compilation import (
    CompilationDiagnostic,
    CompilationResult,
    DiagnosticSeverity,
)
from cuda_opt_agent.schemas.kernel import KernelSource, KernelVariant

# ── Diagnostic regex ───────────────────────────────────────────

# Standard: file(line): severity: message
_DIAG_RE = re.compile(
    r"^(.+?)\((\d+)\):\s*(error|warning|remark|info)(\s+#[\w-]+)?:\s*(.+)",
    re.MULTILINE,
)

# Fatal: "nvcc fatal   : ..." or "ptxas fatal   : ..."
_FATAL_RE = re.compile(
    r"^(nvcc|ptxas)\s+fatal\s*:\s*(.+)",
    re.MULTILINE,
)

# Linker: "/usr/bin/ld: ..."
_LINKER_RE = re.compile(
    r"^(/usr/bin/ld|ld)\s*:\s*(.+)",
    re.MULTILINE,
)


class NVCCRunner:
    """Compile CUDA source via nvcc subprocess.

    Four compilation modes:
    - compile(): source -> cubin (binary kernel object)
    - extract_ptx(): source -> PTX text string (raises on failure)
    - compile_ptx(): source -> CompilationResult with PTX metadata
    - compile_executable(): source -> runnable binary (for ncu profiling)
    """

    def __init__(self, config: CompilerConfig) -> None:
        self._nvcc_path = config.nvcc_path
        self._default_arch = config.default_arch
        self._extra_flags = list(config.extra_flags)

    # ── Public API ─────────────────────────────────────────────

    async def compile(
        self,
        kernel: KernelSource | KernelVariant,
        arch: str | None = None,
        output_dir: str | None = None,
        extra_flags: list[str] | None = None,
    ) -> CompilationResult:
        """Compile kernel source to cubin.

        Sets CompilationResult.cubin_path on success.
        binary_path is NOT set (it is reserved for executables).
        """
        arch = arch or self._default_arch

        with self._prepare_source(kernel, output_dir) as (source_path, work_dir):
            out_path = str(Path(work_dir) / "kernel.cubin")
            flags = [
                *cubin_arch_flag(arch),
                *self._build_flags(extra_flags),
            ]
            return await self._run_nvcc(
                flags=flags,
                source_path=source_path,
                output_path=out_path,
                source_code=kernel.source_code,
                arch=arch,
            )

    async def extract_ptx(
        self,
        kernel: KernelSource | KernelVariant,
        arch: str | None = None,
    ) -> str:
        """Compile kernel source and return PTX text.

        Returns:
            PTX source string.

        Raises:
            PTXExtractionError: If compilation fails or PTX not produced.
        """
        result = await self.compile_ptx(kernel, arch=arch)
        if not result.success:
            err = "; ".join(d.message for d in result.diagnostics[:3]) or result.stderr[:500]
            raise PTXExtractionError(
                f"PTX extraction failed: {err}",
                stderr=result.stderr,
                returncode=result.returncode,
            )
        if not result.ptx_source:
            raise PTXExtractionError(
                "nvcc succeeded but no PTX output was produced",
                stderr=result.stderr,
            )
        return result.ptx_source

    async def compile_ptx(
        self,
        kernel: KernelSource | KernelVariant,
        arch: str | None = None,
        output_dir: str | None = None,
    ) -> CompilationResult:
        """Compile kernel source to PTX and return full CompilationResult.

        Returns:
            CompilationResult with ptx_source and ptx_path populated on success.
        """
        arch = arch or self._default_arch

        with self._prepare_source(kernel, output_dir) as (source_path, work_dir):
            out_path = str(Path(work_dir) / "kernel.ptx")
            flags = [
                *ptx_arch_flag(arch),
                *self._build_flags(),
            ]
            result = await self._run_nvcc(
                flags=flags,
                source_path=source_path,
                output_path=out_path,
                source_code=kernel.source_code,
                arch=arch,
            )

            if result.success and Path(out_path).is_file():
                result.ptx_source = Path(out_path).read_text(errors="replace")
                result.ptx_path = out_path

            return result

    async def compile_executable(
        self,
        kernel: KernelSource | KernelVariant,
        arch: str | None = None,
        output_dir: str | None = None,
        extra_flags: list[str] | None = None,
    ) -> CompilationResult:
        """Compile kernel source to a runnable executable (for ncu profiling).

        Sets CompilationResult.binary_path on success.
        cubin_path is NOT set — the output is an executable, not a cubin.
        """
        arch = arch or self._default_arch

        with self._prepare_source(kernel, output_dir) as (source_path, work_dir):
            out_path = str(Path(work_dir) / "kernel_exec")
            flags = [
                *executable_arch_flag(arch),
                *self._build_flags(extra_flags),
            ]
            result = await self._run_nvcc(
                flags=flags,
                source_path=source_path,
                output_path=out_path,
                source_code=kernel.source_code,
                arch=arch,
            )

            # Promote cubin_path to binary_path for executables
            if result.cubin_path:
                result.binary_path = result.cubin_path
                result.cubin_path = None

            return result

    async def is_available(self) -> bool:
        """Check if nvcc is reachable and functional."""
        if self._nvcc_path == "nvcc" and shutil.which("nvcc") is None:
            return False
        if self._nvcc_path != "nvcc" and not Path(self._nvcc_path).is_file():
            return False
        if os.name == "nt" and shutil.which("cl.exe") is None:
            return False
        try:
            proc = await asyncio.create_subprocess_exec(
                self._nvcc_path, "--version",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await proc.wait()
            return proc.returncode == 0
        except FileNotFoundError:
            return False

    # ── Internal ───────────────────────────────────────────────

    def _build_flags(self, extra_flags: list[str] | None = None) -> list[str]:
        """Assemble common + extra + per-call flags."""
        merged_extra = list(self._extra_flags) + (extra_flags or [])
        return common_flags(extra=merged_extra)

    async def _run_nvcc(
        self,
        flags: list[str],
        source_path: str,
        output_path: str,
        source_code: str,
        arch: str,
    ) -> CompilationResult:
        """Execute nvcc with given flags and parse the result."""
        cmd = [self._nvcc_path, *flags, "-o", output_path, source_path]

        t0 = time.monotonic()
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=120
            )
        except FileNotFoundError:
            return CompilationResult(
                success=False,
                source_code=source_code,
                arch=arch,
                stderr=f"nvcc not found at '{self._nvcc_path}'",
                returncode=-1,
                diagnostics=[CompilationDiagnostic(
                    severity=DiagnosticSeverity.ERROR,
                    message=f"nvcc not found at '{self._nvcc_path}'",
                )],
            )
        except asyncio.TimeoutError:
            return CompilationResult(
                success=False,
                source_code=source_code,
                arch=arch,
                stderr="Compilation timed out after 120 seconds",
                returncode=-1,
                diagnostics=[CompilationDiagnostic(
                    severity=DiagnosticSeverity.ERROR,
                    message="Compilation timed out after 120 seconds",
                )],
            )

        elapsed = time.monotonic() - t0
        stdout_str = stdout.decode(errors="replace")
        stderr_str = stderr.decode(errors="replace")
        success = proc.returncode == 0
        diagnostics = parse_diagnostics("\n".join([stderr_str, stdout_str]))

        output_exists = Path(output_path).is_file()
        binary_path = output_path if success and output_exists else None

        return CompilationResult(
            success=success,
            source_code=source_code,
            arch=arch,
            cubin_path=binary_path,
            diagnostics=diagnostics,
            compile_time_s=elapsed,
            stdout=stdout_str,
            stderr=stderr_str,
            returncode=proc.returncode or 0,
        )

    @staticmethod
    @contextmanager
    def _prepare_source(
        kernel: KernelSource | KernelVariant,
        output_dir: str | None,
    ) -> Generator[tuple[str, str], None, None]:
        """Write kernel source to disk and yield (source_path, work_dir).

        Temp dirs are cleaned up on exit. Explicit output_dir dirs persist.
        """
        if output_dir:
            work_dir = Path(output_dir)
            work_dir.mkdir(parents=True, exist_ok=True)
            source_path = work_dir / "kernel.cu"
            source_path.write_text(kernel.source_code)
            yield str(source_path), str(work_dir)
        else:
            work_dir = Path(tempfile.mkdtemp(prefix="cuda_opt_"))
            source_path = work_dir / "kernel.cu"
            source_path.write_text(kernel.source_code)
            try:
                yield str(source_path), str(work_dir)
            finally:
                shutil.rmtree(work_dir, ignore_errors=True)


# ── Diagnostic Parsing ─────────────────────────────────────────

def parse_diagnostics(stderr: str) -> list[CompilationDiagnostic]:
    """Parse nvcc stderr into structured CompilationDiagnostic objects.

    Handles three nvcc output patterns:
    1. Standard: file(line): error/warning: message
    2. Fatal: nvcc fatal : message  /  ptxas fatal : message
    3. Linker: /usr/bin/ld: message
    """
    diagnostics: list[CompilationDiagnostic] = []
    seen: set[str] = set()

    # 1. Standard diagnostics
    for m in _DIAG_RE.finditer(stderr):
        file_path = m.group(1).strip()
        line = int(m.group(2))
        severity_str = m.group(3).lower()
        message = m.group(5).strip()

        severity = {
            "error": DiagnosticSeverity.ERROR,
            "warning": DiagnosticSeverity.WARNING,
            "remark": DiagnosticSeverity.REMARK,
            "info": DiagnosticSeverity.INFO,
        }.get(severity_str, DiagnosticSeverity.INFO)

        key = f"{file_path}:{line}:{message}"
        if key in seen:
            continue
        seen.add(key)

        diagnostics.append(CompilationDiagnostic(
            severity=severity,
            message=message,
            file=file_path,
            line=line,
            raw=m.group(0),
        ))

    # 2. Fatal errors (nvcc/ptxas)
    for m in _FATAL_RE.finditer(stderr):
        raw = m.group(0).strip()
        if raw in seen:
            continue
        seen.add(raw)
        diagnostics.append(CompilationDiagnostic(
            severity=DiagnosticSeverity.ERROR,
            message=m.group(2).strip(),
            raw=raw,
        ))

    # 3. Linker errors
    for m in _LINKER_RE.finditer(stderr):
        raw = m.group(0).strip()
        if raw in seen:
            continue
        seen.add(raw)
        diagnostics.append(CompilationDiagnostic(
            severity=DiagnosticSeverity.ERROR,
            message=m.group(2).strip(),
            raw=raw,
        ))

    return diagnostics
