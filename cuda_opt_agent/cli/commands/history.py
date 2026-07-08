"""`cuda-opt history` command — show optimization history."""

from __future__ import annotations

import asyncio
from pathlib import Path

import typer

from cuda_opt_agent.cli.output import print_error, print_info, print_run_summary
from cuda_opt_agent.core.config import load_settings


def history_cmd(
    kernel: str = typer.Option(None, "--kernel", "-k", help="Filter by kernel name"),
    limit: int = typer.Option(20, "--limit", "-n", help="Max runs to show"),
    config: Path = typer.Option(None, "--config", "-c", help="Path to config.yaml"),
):
    """Show optimization history from the database."""
    settings = load_settings(config)

    results = asyncio.run(_query_history(settings, kernel_name=kernel, limit=limit))

    if not results:
        print_info("No optimization runs found.")
        return

    for run_data in results:
        _print_history_entry(run_data)
        print()


async def _query_history(settings, kernel_name: str | None = None, limit: int = 20):
    from cuda_opt_agent.memory.database import init_db
    from cuda_opt_agent.memory.repositories.run_repo import RunRepo
    from cuda_opt_agent.memory.database import get_session

    try:
        init_db(settings.database)
    except Exception:
        pass

    with get_session() as session:
        repo = RunRepo(session)
        runs = repo.list_runs(offset=0, limit=limit)

    if kernel_name:
        runs = [r for r in runs if r.kernel.name == kernel_name]

    return runs


def _print_history_entry(run) -> None:
    best = f"{run.best_speedup:.3f}x" if run.best_speedup > 1.0 else "none"
    print_info(f"Run: {run.run_id}")
    print_info(f"  Kernel:     {run.kernel.name}")
    print_info(f"  Status:     {run.status}")
    print_info(f"  Iterations: {run.iteration_count}")
    print_info(f"  Best:       {best} ({run.best_variant_id or '—'})")
