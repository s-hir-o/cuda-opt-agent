"""Tests for the interactive CLI shell."""

from typer.testing import CliRunner

from cuda_opt_agent.cli.app import _split_shell_args, app

runner = CliRunner()


def test_shell_runs_help_and_exits() -> None:
    result = runner.invoke(app, ["shell"], input="help\nexit\n")

    assert result.exit_code == 0
    assert "CUDA optimization shell" in result.output
    assert "benchmark" in result.output
    assert "profile" in result.output


def test_shell_recovers_from_unknown_command() -> None:
    result = runner.invoke(app, ["shell"], input="not-a-command\nexit\n")

    assert result.exit_code == 0
    assert "No such command 'not-a-command'" in result.output


def test_shell_rejects_nested_session() -> None:
    result = runner.invoke(app, ["shell"], input="shell\nexit\n")

    assert result.exit_code == 0
    assert "Already inside" in result.output


def test_shell_preserves_windows_relative_path() -> None:
    args = _split_shell_args(
        r"benchmark .\workspace\matrixMul_cli_harness.cu --arch sm_75"
    )

    assert args == [
        "benchmark",
        r".\workspace\matrixMul_cli_harness.cu",
        "--arch",
        "sm_75",
    ]


def test_shell_parses_quoted_windows_path() -> None:
    args = _split_shell_args(
        r'benchmark "D:\work\repo\cuda samples\kernel.cu" --arch sm_75'
    )

    assert args == [
        "benchmark",
        r"D:\work\repo\cuda samples\kernel.cu",
        "--arch",
        "sm_75",
    ]


def test_benchmark_exposes_include_directory_option() -> None:
    result = runner.invoke(app, ["benchmark", "--help"])

    assert result.exit_code == 0
    assert "--include-dir" in result.output
