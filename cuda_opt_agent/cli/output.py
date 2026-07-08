"""Rich output formatting for CLI commands.

Uses the Rich library for colored tables, progress bars, and panels.
Falls back to plain text if Rich is not available.
"""

from __future__ import annotations

from typing import Any

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    _HAS_RICH = True
except ImportError:
    _HAS_RICH = False

from cuda_opt_agent.schemas.optimization import OptimizationRun


console = Console() if _HAS_RICH else None


def print_run_summary(run: OptimizationRun) -> None:
    """Print a summary of a completed optimization run."""
    if _HAS_RICH:
        _rich_run_summary(run)
    else:
        _plain_run_summary(run)


def print_iteration_table(run: OptimizationRun) -> None:
    """Print a table of all iterations in the run."""
    if _HAS_RICH:
        _rich_iteration_table(run)
    else:
        _plain_iteration_table(run)


def print_error(msg: str) -> None:
    """Print an error message."""
    if _HAS_RICH:
        console.print(f"[bold red]Error:[/bold red] {msg}")
    else:
        print(f"Error: {msg}")


def print_info(msg: str) -> None:
    """Print an informational message."""
    if _HAS_RICH:
        console.print(f"[dim]{msg}[/dim]")
    else:
        print(msg)


def print_success(msg: str) -> None:
    """Print a success message."""
    if _HAS_RICH:
        console.print(f"[bold green]{msg}[/bold green]")
    else:
        print(msg)


# ── Rich formatting ────────────────────────────────────────────

def _rich_run_summary(run: OptimizationRun) -> None:
    best = f"{run.best_speedup:.3f}x" if run.best_speedup > 1.0 else "none"

    table = Table(title="Optimization Run Summary", show_header=False)
    table.add_column("Key", style="bold")
    table.add_column("Value")
    table.add_row("Run ID", run.run_id)
    table.add_row("Kernel", run.kernel.name)
    table.add_row("Architecture", run.target_arch)
    table.add_row("Status", _status_style(run.status))
    table.add_row("Iterations", str(run.iteration_count))
    table.add_row("Best Speedup", best)
    table.add_row("Best Variant", run.best_variant_id or "—")

    console.print(table)


def _rich_iteration_table(run: OptimizationRun) -> None:
    table = Table(title=f"Iterations — {run.run_id}")
    table.add_column("#", justify="right")
    table.add_column("Variant")
    table.add_column("Speedup", justify="right")
    table.add_column("Bottleneck")
    table.add_column("Transform")
    table.add_column("Status")

    for it in run.iterations:
        speedup_str = f"{it.achieved_speedup:.3f}x" if it.speedup else "—"
        bottleneck = it.bottleneck.primary.value if it.bottleneck else "—"
        transform = str(it.transform_applied) if it.transform_applied else "—"
        status = "[red]error[/red]" if it.error else "[green]ok[/green]"

        table.add_row(
            str(it.iteration),
            it.variant_id,
            speedup_str,
            bottleneck,
            transform,
            status,
        )

    console.print(table)


def _status_style(status: str) -> str:
    styles = {
        "running": "[yellow]running[/yellow]",
        "converged": "[green]converged[/green]",
        "completed": "[green]completed[/green]",
        "failed": "[red]failed[/red]",
    }
    return styles.get(status, status)


# ── Plain text fallback ────────────────────────────────────────

def _plain_run_summary(run: OptimizationRun) -> None:
    print(f"Run: {run.run_id}")
    print(f"Kernel: {run.kernel.name}")
    print(f"Arch: {run.target_arch}")
    print(f"Status: {run.status}")
    print(f"Iterations: {run.iteration_count}")
    print(f"Best Speedup: {run.best_speedup:.3f}x")
    print(f"Best Variant: {run.best_variant_id or 'none'}")


def _plain_iteration_table(run: OptimizationRun) -> None:
    print(f"\n{'#':>4} {'Variant':<20} {'Speedup':>10} {'Bottleneck':<18} {'Status':<8}")
    print("-" * 65)
    for it in run.iterations:
        speedup = f"{it.achieved_speedup:.3f}x" if it.speedup else "—"
        bottleneck = it.bottleneck.primary.value if it.bottleneck else "—"
        status = "error" if it.error else "ok"
        print(f"{it.iteration:>4} {it.variant_id:<20} {speedup:>10} {bottleneck:<18} {status:<8}")
