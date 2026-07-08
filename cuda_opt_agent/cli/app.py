"""Typer CLI application root.

Commands:
    cuda-opt optimize <file>    Run full optimization pipeline
    cuda-opt profile <file>     Profile a compiled kernel
    cuda-opt benchmark <file>   Benchmark a kernel variant
    cuda-opt history            Show optimization history
"""

from __future__ import annotations

import typer

from cuda_opt_agent.cli.commands.optimize import optimize_cmd
from cuda_opt_agent.cli.commands.profile import profile_cmd
from cuda_opt_agent.cli.commands.benchmark import benchmark_cmd
from cuda_opt_agent.cli.commands.history import history_cmd

app = typer.Typer(
    name="cuda-opt",
    help="CUDA kernel auto-optimization agent.",
    no_args_is_help=True,
)

app.command(name="optimize", help="Run full optimization on a CUDA kernel.")(optimize_cmd)
app.command(name="profile", help="Profile a CUDA kernel with NCU.")(profile_cmd)
app.command(name="benchmark", help="Benchmark a CUDA kernel variant.")(benchmark_cmd)
app.command(name="history", help="Show optimization history.")(history_cmd)


def main():
    app()


if __name__ == "__main__":
    main()
