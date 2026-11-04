"""Tests for Windows MSVC environment parsing and discovery."""

from pathlib import Path
from types import SimpleNamespace

from cuda_opt_agent.build import windows_toolchain


def test_ensure_msvc_environment_loads_developer_path(monkeypatch, tmp_path: Path) -> None:
    vsdevcmd = tmp_path / "VsDevCmd.bat"
    vsdevcmd.write_text("@echo off\n")
    original_path = "C:\\Windows\\System32"
    developer_path = "C:\\VisualStudio\\VC\\bin"

    monkeypatch.setattr(windows_toolchain.os, "name", "nt")
    monkeypatch.setattr(windows_toolchain, "_find_vsdevcmd", lambda: vsdevcmd)
    monkeypatch.setattr(
        windows_toolchain.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout=f"Path={developer_path};{original_path}\r\nVSCMD_ARG_TGT_ARCH=x64\r\n".encode(),
        ),
    )

    def fake_which(name: str) -> str | None:
        if name == "cl.exe" and windows_toolchain.os.environ.get("Path", "").startswith(
            developer_path
        ):
            return f"{developer_path}\\cl.exe"
        return None

    monkeypatch.setattr(windows_toolchain.shutil, "which", fake_which)

    assert windows_toolchain.ensure_windows_msvc_environment() is True
    assert windows_toolchain.os.environ["VSCMD_ARG_TGT_ARCH"] == "x64"


def test_ensure_msvc_environment_fails_without_visual_studio(monkeypatch) -> None:
    monkeypatch.setattr(windows_toolchain.os, "name", "nt")
    monkeypatch.setattr(windows_toolchain.shutil, "which", lambda name: None)
    monkeypatch.setattr(windows_toolchain, "_find_vsdevcmd", lambda: None)

    assert windows_toolchain.ensure_windows_msvc_environment() is False


def test_ensure_msvc_environment_prefers_developer_path_when_cmd_prints_both_cases(
    monkeypatch, tmp_path: Path
) -> None:
    vsdevcmd = tmp_path / "VsDevCmd.bat"
    vsdevcmd.write_text("@echo off\n")
    developer_path = "C:\\VisualStudio\\VC\\bin"
    stale_path = "C:\\Windows\\System32"

    monkeypatch.setattr(windows_toolchain.os, "name", "nt")
    monkeypatch.setattr(windows_toolchain, "_find_vsdevcmd", lambda: vsdevcmd)
    monkeypatch.setattr(
        windows_toolchain.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout=f"PATH={developer_path}\r\nPath={stale_path}\r\n".encode(),
        ),
    )
    monkeypatch.setattr(
        windows_toolchain.shutil,
        "which",
        lambda name: developer_path if name == "cl.exe" and
        windows_toolchain.os.environ["PATH"] == developer_path else None,
    )

    assert windows_toolchain.ensure_windows_msvc_environment() is True
    assert windows_toolchain.os.environ["PATH"] == developer_path
