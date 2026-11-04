"""End-to-end CLI behavior tests for optimize output emission."""

from __future__ import annotations

from typer.testing import CliRunner

import cuda_opt_agent.cli.commands.optimize as optimize_module
from cuda_opt_agent.cli.app import app
from cuda_opt_agent.schemas.benchmark import SpeedupResult
from cuda_opt_agent.schemas.kernel import KernelSource, KernelVariant
from cuda_opt_agent.schemas.optimization import OptimizationIteration, OptimizationRun


runner = CliRunner()


def test_optimize_writes_best_variant_source(monkeypatch, tmp_path):
    source = """\
__global__ void k(float *out) {
    out[0] = 1.0f;
}

int main(int argc, char **argv) {
    const char *flag = "--output";
    return flag[0] == argv[0][0] && argc > 0 ? 0 : 0;
}
"""
    optimized_source = source.replace("1.0f", "2.0f") + "\n// optimized\n"
    input_path = tmp_path / "kernel.cu"
    output_path = tmp_path / "optimized.cu"
    input_path.write_text(source)

    captured: dict[str, object] = {}

    async def fake_run_optimize(kernel: KernelSource, settings):
        captured["entry_point"] = kernel.entry_point
        captured["arch"] = settings.compiler.default_arch
        captured["max_iterations"] = settings.optimization.max_iterations

        run = OptimizationRun(
            run_id="run_e2e",
            kernel=kernel,
            target_arch=settings.compiler.default_arch,
            max_iterations=settings.optimization.max_iterations,
        )
        baseline = KernelVariant(
            variant_id="baseline",
            parent_source=kernel,
            source_code=kernel.source_code,
            entry_point=kernel.entry_point,
            iteration=0,
        )
        optimized = KernelVariant(
            variant_id="optimized",
            parent_source=kernel,
            source_code=optimized_source,
            entry_point=kernel.entry_point,
            iteration=1,
        )
        run.add_iteration(
            OptimizationIteration(
                iteration=0,
                variant_id="baseline",
                variant=baseline,
            )
        )
        run.add_iteration(
            OptimizationIteration(
                iteration=1,
                variant_id="optimized",
                variant=optimized,
                speedup=SpeedupResult(
                    baseline_variant_id="baseline",
                    candidate_variant_id="optimized",
                    baseline_time_ms=2.0,
                    candidate_time_ms=1.0,
                    speedup=2.0,
                    significant=True,
                    effective_speedup=2.0,
                ),
            )
        )
        return run

    monkeypatch.setattr(optimize_module, "_run_optimize", fake_run_optimize)

    result = runner.invoke(
        app,
        [
            "optimize",
            str(input_path),
            "--entry-point",
            "k",
            "--arch",
            "sm_80",
            "--max-iter",
            "1",
            "--output",
            str(output_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert output_path.read_text() == optimized_source
    assert captured == {
        "entry_point": "k",
        "arch": "sm_80",
        "max_iterations": 1,
    }
    assert "Optimized source written" in result.output


def test_optimize_writes_baseline_when_no_faster_variant(monkeypatch, tmp_path):
    source = """\
__global__ void k(float *out) {
    out[0] = 1.0f;
}

int main(int argc, char **argv) {
    const char *flag = "--output";
    return flag[0] == argv[0][0] && argc > 0 ? 0 : 0;
}
"""
    input_path = tmp_path / "kernel.cu"
    output_path = tmp_path / "optimized.cu"
    input_path.write_text(source)

    async def fake_run_optimize(kernel: KernelSource, settings):
        run = OptimizationRun(
            run_id="run_baseline",
            kernel=kernel,
            target_arch=settings.compiler.default_arch,
            max_iterations=settings.optimization.max_iterations,
        )
        baseline = KernelVariant(
            variant_id="baseline",
            parent_source=kernel,
            source_code=kernel.source_code,
            entry_point=kernel.entry_point,
            iteration=0,
        )
        run.add_iteration(
            OptimizationIteration(
                iteration=0,
                variant_id="baseline",
                variant=baseline,
            )
        )
        return run

    monkeypatch.setattr(optimize_module, "_run_optimize", fake_run_optimize)

    result = runner.invoke(
        app,
        [
            "optimize",
            str(input_path),
            "--entry-point",
            "k",
            "--output",
            str(output_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert output_path.read_text() == source
    assert "No faster variant was found" in result.output
    assert "Optimized source written" in result.output
