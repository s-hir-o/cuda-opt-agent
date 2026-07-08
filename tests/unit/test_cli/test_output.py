"""Tests for CLI output formatting."""

import pytest

from cuda_opt_agent.cli.output import (
    print_run_summary,
    print_iteration_table,
    print_error,
    print_info,
    print_success,
)
from cuda_opt_agent.schemas.bottleneck import BottleneckClass, BottleneckResult
from cuda_opt_agent.schemas.kernel import KernelLanguage, KernelSource, KernelVariant
from cuda_opt_agent.schemas.optimization import OptimizationIteration, OptimizationRun


KERNEL = KernelSource(name="test.cu", language=KernelLanguage.CUDA, source_code="")


class TestOutput:

    def test_print_run_summary_no_crash(self, capsys):
        run = OptimizationRun(
            run_id="run_001", kernel=KERNEL, target_arch="sm_80",
            status="converged", best_speedup=2.5, best_variant_id="v3",
        )
        print_run_summary(run)
        output = capsys.readouterr().out
        assert "run_001" in output or "test.cu" in output  # either rich or plain

    def test_print_iteration_table_no_crash(self, capsys):
        variant = KernelVariant(
            variant_id="v1", parent_source=KERNEL, source_code="// opt", iteration=1,
        )
        iteration = OptimizationIteration(
            iteration=1,
            variant_id="v1",
            variant=variant,
            bottleneck=BottleneckResult(primary=BottleneckClass.MEMORY_BOUND, confidence=0.9),
            speedup=None,
        )
        run = OptimizationRun(
            run_id="run_001", kernel=KERNEL, target_arch="sm_80",
            iterations=[iteration],
        )
        print_iteration_table(run)
        output = capsys.readouterr().out
        assert "v1" in output

    def test_print_error(self, capsys):
        print_error("something broke")
        output = capsys.readouterr().out
        assert "something broke" in output

    def test_print_info(self, capsys):
        print_info("informational")
        output = capsys.readouterr().out
        assert "informational" in output

    def test_print_success(self, capsys):
        print_success("all good")
        output = capsys.readouterr().out
        assert "all good" in output
