"""Typer CLI application root.

Commands:
    cuda-opt optimize <file>    Run full optimization pipeline
    cuda-opt profile <file>     Profile a compiled kernel
    cuda-opt benchmark <file>   Benchmark a kernel variant
    cuda-opt history            Show optimization history
"""

from __future__ import annotations

import os
import shlex

import typer

from cuda_opt_agent.cli.commands.benchmark import benchmark_cmd
from cuda_opt_agent.cli.commands.history import history_cmd
from cuda_opt_agent.cli.commands.optimize import optimize_cmd
from cuda_opt_agent.cli.commands.profile import profile_cmd

app = typer.Typer(
    name="cuda-opt",
    help="CUDA kernel auto-optimization agent.",
    no_args_is_help=True,
)

app.command(name="optimize", help="Run full optimization on a CUDA kernel.")(optimize_cmd)
app.command(name="profile", help="Profile a CUDA kernel with NCU.")(profile_cmd)
app.command(name="benchmark", help="Benchmark a CUDA kernel variant.")(benchmark_cmd)
app.command(name="history", help="Show optimization history.")(history_cmd)


def _split_shell_args(command_line: str) -> list[str]:
    """Split interactive input without consuming Windows path separators."""
    if os.name != "nt":
        return shlex.split(command_line)

    lexer = shlex.shlex(command_line, posix=False)
    lexer.whitespace_split = True
    lexer.commenters = ""
    return [
        arg[1:-1] if len(arg) >= 2 and arg[0] == arg[-1] and arg[0] in "\"'" else arg
        for arg in lexer
    ]


@app.command(name="shell", help="Open an interactive CUDA optimization command shell.")
def shell_cmd() -> None:
    """Run CLI commands repeatedly without retyping the executable name."""
    typer.echo("CUDA optimization shell. Type 'help' for commands and 'exit' to quit.")
    command = typer.main.get_command(app)

    while True:
        try:
            command_line = typer.prompt("cuda-opt", prompt_suffix="> ").strip()
        except (typer.Abort, EOFError):
            typer.echo()
            return

        if not command_line:
            continue
        if command_line.lower() in {"exit", "quit"}:
            return

        args = _split_shell_args(command_line)
        if args == ["help"]:
            args = ["--help"]
        elif args and args[0] == "shell":
            typer.echo("Already inside the CUDA optimization shell.", err=True)
            continue

        try:
            command.main(args=args, prog_name="cuda-opt", standalone_mode=False)
        except SystemExit:
            # Typer's rich usage-error renderer can still raise SystemExit.
            continue
        except typer.Abort:
            typer.echo("Aborted.", err=True)
        except Exception as exc:
            show_error = getattr(exc, "show", None)
            if callable(show_error):
                show_error()
            else:
                typer.echo(f"Command failed: {exc}", err=True)


def main() -> None:
    from cuda_opt_agent.build.windows_toolchain import ensure_windows_msvc_environment

    ensure_windows_msvc_environment()
    app()


if __name__ == "__main__":
    main()
