"""Windows host-compiler environment discovery for NVCC."""

from __future__ import annotations

import locale
import os
import shutil
import subprocess
from pathlib import Path


def ensure_windows_msvc_environment() -> bool:
    """Load a Visual Studio developer environment when cl.exe is not on PATH."""
    if os.name != "nt" or shutil.which("cl.exe") is not None:
        return True

    vsdevcmd = _find_vsdevcmd()
    if vsdevcmd is None:
        return False

    command = f'call "{vsdevcmd}" -arch=x64 -host_arch=x64 >nul && set'
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            check=False,
            executable=os.environ.get("COMSPEC", "cmd.exe"),
            shell=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False

    if result.returncode != 0:
        return False

    encoding = locale.getpreferredencoding(False)
    developer_path: str | None = None
    for line in result.stdout.decode(encoding, errors="replace").splitlines():
        key, separator, value = line.partition("=")
        if separator and key and not key.startswith("="):
            if key.upper() == "PATH":
                # cmd.exe can print both PATH (set by VsDevCmd) and a stale Path.
                # Preserve the developer path regardless of their output order.
                if developer_path is None or key == "PATH":
                    developer_path = value
                continue
            os.environ[key] = value

    if developer_path is not None:
        os.environ["PATH"] = developer_path

    return shutil.which("cl.exe") is not None


def _find_vsdevcmd() -> Path | None:
    install_dir = os.environ.get("VSINSTALLDIR")
    if install_dir:
        candidate = Path(install_dir) / "Common7" / "Tools" / "VsDevCmd.bat"
        if candidate.is_file():
            return candidate

    vswhere = _find_vswhere()
    if vswhere is not None:
        try:
            result = subprocess.run(
                [
                    str(vswhere),
                    "-latest",
                    "-products",
                    "*",
                    "-requires",
                    "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                    "-property",
                    "installationPath",
                ],
                capture_output=True,
                check=False,
                text=True,
                timeout=15,
            )
        except (OSError, subprocess.TimeoutExpired):
            result = None

        if result is not None and result.returncode == 0:
            installation = result.stdout.strip()
            candidate = Path(installation) / "Common7" / "Tools" / "VsDevCmd.bat"
            if installation and candidate.is_file():
                return candidate

    candidates: list[Path] = []
    for variable in ("ProgramFiles", "ProgramW6432", "ProgramFiles(x86)"):
        root_value = os.environ.get(variable)
        if not root_value:
            continue
        root = Path(root_value) / "Microsoft Visual Studio"
        candidates.extend(root.glob("*/*/Common7/Tools/VsDevCmd.bat"))

    return max(candidates, key=lambda path: path.stat().st_mtime, default=None)


def _find_vswhere() -> Path | None:
    for variable in ("ProgramFiles(x86)", "ProgramFiles"):
        root_value = os.environ.get(variable)
        if not root_value:
            continue
        candidate = (
            Path(root_value)
            / "Microsoft Visual Studio"
            / "Installer"
            / "vswhere.exe"
        )
        if candidate.is_file():
            return candidate
    return None
