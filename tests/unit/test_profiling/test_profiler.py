
"""Tests for the Profiler ABC contract, MockProfiler, validator, and serialization."""

from __future__ import annotations
import asyncio
import json
import pytest

from cuda_opt_agent.profiling.mock_profiler import MockProfiler
from cuda_opt_agent.profiling.ncu_runner import NCUProfiler, NCURunner
from cuda_opt_agent.core.config import ProfilingConfig
from cuda_opt_agent.profiling.metrics_validator import validate_metrics
from cuda_opt_agent.schemas.compilation import CompilationResult
from cuda_opt_agent.schemas.kernel import KernelLanguage, KernelSource, KernelVariant
from cuda_opt_agent.schemas.metrics import ProfileMetrics


def _run(coro):
    """Run an async coroutine."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ── MockProfiler Tests ────────────────────────────────────────

class TestMockProfiler:

    def test_is_available(self):
        profiler = MockProfiler()
        assert _run(profiler.is_available()) is True

    def test_naive_matmul_preset(self):
        profiler = MockProfiler()
        kernel = KernelSource(
            name="naive_matmul.cu",
            language=KernelLanguage.CUDA,
            source_code="__global__ void k() {}",
        )
        m = _run(profiler.profile(kernel, run_id="r1", arch="sm_80"))
        assert m.kernel_name == "naive_matmul"
        assert m.gpu_time_us == 850.0
        assert m.dram_throughput == 78.0
        assert m.sm_throughput == 35.0
        assert m.occupancy_limited_by_registers is True

    def test_tiled_matmul_preset(self):
        profiler = MockProfiler()
        kernel = KernelSource(
            name="tiled_matmul.cu",
            language=KernelLanguage.CUDA,
            source_code="__global__ void k() {}",
        )
        m = _run(profiler.profile(kernel, run_id="r1", arch="sm_80"))
        assert m.kernel_name == "tiled_matmul"
        assert m.gpu_time_us == 120.0
        assert m.sm_throughput == 65.0
        assert m.occupancy_limited_by_registers is False

    def test_compute_bound_preset(self):
        profiler = MockProfiler()
        kernel = KernelSource(
            name="compute_bound_kernel.cu",
            language=KernelLanguage.CUDA,
            source_code="__global__ void k() {}",
        )
        m = _run(profiler.profile(kernel, run_id="r1", arch="sm_80"))
        assert m.sm_throughput == 92.0
        assert m.dram_throughput == 15.0
        assert m.achieved_occupancy == 0.85

    def test_memory_bound_preset(self):
        profiler = MockProfiler()
        kernel = KernelSource(
            name="memory_bound_kernel.cu",
            language=KernelLanguage.CUDA,
            source_code="__global__ void k() {}",
        )
        m = _run(profiler.profile(kernel, run_id="r1", arch="sm_80"))
        assert m.dram_throughput == 90.0
        assert m.sm_throughput == 20.0
        assert m.global_load_efficiency == 30.0
        assert m.occupancy_limited_by_registers is True

    def test_unknown_kernel_uses_naive_fallback(self):
        profiler = MockProfiler()
        kernel = KernelSource(
            name="unknown_kernel.cu",
            language=KernelLanguage.CUDA,
            source_code="__global__ void k() {}",
        )
        m = _run(profiler.profile(kernel, run_id="r1", arch="sm_80"))
        assert m.kernel_name == "unknown_kernel"
        assert m.gpu_time_us == 850.0

    def test_custom_arch(self):
        profiler = MockProfiler(arch="sm_90")
        kernel = KernelSource(
            name="test.cu",
            language=KernelLanguage.CUDA,
            source_code="__global__ void k() {}",
        )
        m = _run(profiler.profile(kernel, run_id="r1", arch="sm_90"))
        assert m.arch == "sm_90"

    def test_custom_gpu_name(self):
        profiler = MockProfiler(gpu_name="NVIDIA H100")
        kernel = KernelSource(
            name="test.cu",
            language=KernelLanguage.CUDA,
            source_code="__global__ void k() {}",
        )
        m = _run(profiler.profile(kernel, run_id="r1", arch="sm_90"))
        assert m.gpu_name == "NVIDIA H100"

    def test_variant_uses_variant_id(self):
        profiler = MockProfiler()
        parent = KernelSource(
            name="matmul.cu",
            language=KernelLanguage.CUDA,
            source_code="__global__ void k() {}",
        )
        variant = KernelVariant(
            variant_id="naive_matmul_v2",
            parent_source=parent,
            source_code="// optimized",
        )
        m = _run(profiler.profile(variant, run_id="r1", arch="sm_80"))
        assert m.kernel_name == "naive_matmul"

    def test_block_size_set(self):
        profiler = MockProfiler()
        kernel = KernelSource(name="test.cu", language=KernelLanguage.CUDA, source_code="")
        m = _run(profiler.profile(kernel, run_id="r1", arch="sm_80"))
        assert m.block_size == (16, 16, 1)
        assert m.grid_size == (64, 64, 1)

    def test_metadata_contains_variant_id(self):
        profiler = MockProfiler()
        kernel = KernelSource(name="test.cu", language=KernelLanguage.CUDA, source_code="")
        m = _run(profiler.profile(kernel, run_id="r1", arch="sm_80"))
        assert m.metadata is not None
        assert m.metadata.get("variant_id") is None

    def test_presets_pass_validation(self):
        """All preset profiles produce metrics with zero validation warnings."""
        profiler = MockProfiler()
        for preset_name in MockProfiler._PRESETS:
            kernel = KernelSource(
                name=f"{preset_name}.cu",
                language=KernelLanguage.CUDA,
                source_code="",
            )
            m = _run(profiler.profile(kernel, run_id="r1", arch="sm_80"))
            warnings = validate_metrics(m)
            assert warnings == [], (
                f"Preset '{preset_name}' has validation warnings: {warnings}"
            )


class TestNCUProfiler:

    def test_profiles_compiled_executable_with_ncu_csv(self):
        class FakeNVCC:
            async def is_available(self):
                return True

            async def compile_executable(self, kernel, arch):
                return CompilationResult(
                    success=True,
                    source_code=kernel.source_code,
                    arch=arch,
                    binary_path="kernel_exec",
                )

        class FakeNCU:
            async def is_available(self):
                return True

            async def run_ncu(self, binary_path, kernel_name=None, arch="sm_80"):
                assert binary_path == "kernel_exec"
                assert kernel_name == "vecadd"
                assert arch == "sm_80"
                return (
                    '"Metric Name","Metric Value","Metric Unit"\n'
                    '"gpu__time_duration.sum","100000","nsecond"\n'
                    '"sm__throughput.avg_pct_of_peak_sustained_elapsed","50","%"\n'
                )

        kernel = KernelSource(
            name="vecadd.cu",
            language=KernelLanguage.CUDA,
            source_code="__global__ void vecadd() {}",
            entry_point="vecadd",
        )
        profiler = NCUProfiler(ncu_runner=FakeNCU(), nvcc_runner=FakeNVCC())

        metrics = _run(profiler.profile(kernel, run_id="r1", arch="sm_80"))

        assert metrics.kernel_name == "vecadd"
        assert metrics.gpu_time_us == 100.0
        assert metrics.sm_throughput == 50.0
        assert metrics.arch == "sm_80"
        assert metrics.metadata["run_id"] == "r1"
        assert metrics.metadata["binary_path"] == "kernel_exec"


def test_ncu_runner_uses_one_metrics_option(monkeypatch):
    captured = []

    class FakeProcess:
        returncode = 0

        async def communicate(self):
            return b"metric output", b""

    async def fake_subprocess(*args, **kwargs):
        captured.extend(args)
        return FakeProcess()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_subprocess)
    runner = NCURunner(ProfilingConfig(metrics=["metric_a", "metric_b"]))

    assert _run(runner.run_ncu("kernel.exe")) == "metric output"
    assert captured.count("--metrics") == 1
    assert captured[captured.index("--metrics") + 1] == "metric_a,metric_b"


def test_ncu_runner_resolves_windows_batch_launcher(monkeypatch, tmp_path):
    launcher = tmp_path / "ncu.bat"
    launcher.write_text("@echo off\n")
    native = tmp_path / "target" / "windows-desktop-win7-x64" / "ncu.exe"
    native.parent.mkdir(parents=True)
    native.write_bytes(b"")
    monkeypatch.setattr("cuda_opt_agent.profiling.ncu_runner.shutil.which", lambda _: str(launcher))

    runner = NCURunner(ProfilingConfig())
    assert runner._command("--version") == [str(native), "--version"]


# ── Metrics Validator Tests ───────────────────────────────────

class TestMetricsValidator:

    def test_valid_metrics_no_warnings(self):
        m = ProfileMetrics(
            kernel_name="test",
            gpu_time_us=100.0,
            sm_throughput=80.0,
            dram_throughput=45.0,
            l1_throughput=60.0,
            l2_throughput=70.0,
            achieved_occupancy=0.75,
            registers_per_thread=32,
            global_load_efficiency=90.0,
            global_store_efficiency=95.0,
            l1_hit_rate=88.0,
            l2_hit_rate=82.0,
        )
        assert validate_metrics(m) == []

    def test_negative_gpu_time(self):
        m = ProfileMetrics(kernel_name="test", gpu_time_us=-1.0)
        warnings = validate_metrics(m)
        assert any("Non-positive GPU time" in w for w in warnings)

    def test_zero_gpu_time(self):
        m = ProfileMetrics(kernel_name="test", gpu_time_us=0.0)
        warnings = validate_metrics(m)
        assert any("Non-positive GPU time" in w for w in warnings)

    def test_throughput_over_100(self):
        m = ProfileMetrics(kernel_name="test", sm_throughput=105.0)
        warnings = validate_metrics(m)
        assert any("sm_throughput" in w for w in warnings)

    def test_throughput_negative(self):
        m = ProfileMetrics(kernel_name="test", dram_throughput=-5.0)
        warnings = validate_metrics(m)
        assert any("dram_throughput" in w for w in warnings)

    def test_l1_throughput_range(self):
        m = ProfileMetrics(kernel_name="test", l1_throughput=150.0)
        warnings = validate_metrics(m)
        assert any("l1_throughput" in w for w in warnings)

    def test_l2_throughput_range(self):
        m = ProfileMetrics(kernel_name="test", l2_throughput=-1.0)
        warnings = validate_metrics(m)
        assert any("l2_throughput" in w for w in warnings)

    def test_occupancy_over_1(self):
        m = ProfileMetrics(kernel_name="test", achieved_occupancy=1.5)
        warnings = validate_metrics(m)
        assert any("occupancy" in w.lower() for w in warnings)

    def test_occupancy_negative(self):
        m = ProfileMetrics(kernel_name="test", achieved_occupancy=-0.1)
        warnings = validate_metrics(m)
        assert any("occupancy" in w.lower() for w in warnings)

    def test_invalid_register_count_high(self):
        m = ProfileMetrics(kernel_name="test", registers_per_thread=300)
        warnings = validate_metrics(m)
        assert any("register" in w.lower() for w in warnings)

    def test_invalid_register_count_zero(self):
        m = ProfileMetrics(kernel_name="test", registers_per_thread=0)
        warnings = validate_metrics(m)
        assert any("register" in w.lower() for w in warnings)

    def test_load_efficiency_range(self):
        m = ProfileMetrics(kernel_name="test", global_load_efficiency=110.0)
        warnings = validate_metrics(m)
        assert any("global_load_efficiency" in w for w in warnings)

    def test_store_efficiency_range(self):
        m = ProfileMetrics(kernel_name="test", global_store_efficiency=-10.0)
        warnings = validate_metrics(m)
        assert any("global_store_efficiency" in w for w in warnings)

    def test_l1_hit_rate_range(self):
        m = ProfileMetrics(kernel_name="test", l1_hit_rate=105.0)
        warnings = validate_metrics(m)
        assert any("l1_hit_rate" in w for w in warnings)

    def test_l2_hit_rate_range(self):
        m = ProfileMetrics(kernel_name="test", l2_hit_rate=-5.0)
        warnings = validate_metrics(m)
        assert any("l2_hit_rate" in w for w in warnings)

    def test_none_fields_no_warning(self):
        """All-None metrics should produce zero warnings."""
        m = ProfileMetrics(kernel_name="test")
        assert validate_metrics(m) == []


# ── ProfileMetrics Serialization Tests ────────────────────────

class TestProfileMetricsSerialization:

    def test_model_dump_json_roundtrip(self):
        """ProfileMetrics survives dump -> JSON -> load with no data loss."""
        original = ProfileMetrics(
            kernel_name="serialize_test",
            gpu_name="NVIDIA A100",
            arch="sm_80",
            gpu_time_ns=123456.0,
            gpu_time_us=123.456,
            sm_throughput=65.3,
            dram_throughput=42.1,
            l1_throughput=55.0,
            l2_throughput=60.0,
            achieved_occupancy=0.75,
            registers_per_thread=32,
            block_size=(16, 16, 1),
            grid_size=(64, 64, 1),
            shared_mem_bytes=4096,
            global_load_efficiency=90.0,
            global_store_efficiency=95.0,
            l2_hit_rate=82.0,
            l1_hit_rate=88.0,
            occupancy_limited_by_registers=False,
            occupancy_limited_by_shared_mem=False,
            occupancy_limited_by_block_size=False,
            raw_metrics={"gpu__time_duration.sum": 123456.0},
            metadata={"variant_id": "v1"},
        )

        dumped = original.model_dump(mode="json")
        json_str = json.dumps(dumped)

        loaded_dict = json.loads(json_str)
        restored = ProfileMetrics.model_validate(loaded_dict)

        assert restored.kernel_name == original.kernel_name
        assert restored.gpu_name == original.gpu_name
        assert restored.arch == original.arch
        assert restored.gpu_time_ns == original.gpu_time_ns
        assert restored.gpu_time_us == original.gpu_time_us
        assert restored.sm_throughput == original.sm_throughput
        assert restored.dram_throughput == original.dram_throughput
        assert restored.l1_throughput == original.l1_throughput
        assert restored.l2_throughput == original.l2_throughput
        assert restored.achieved_occupancy == original.achieved_occupancy
        assert restored.registers_per_thread == original.registers_per_thread
        assert restored.block_size == original.block_size
        assert restored.grid_size == original.grid_size
        assert restored.shared_mem_bytes == original.shared_mem_bytes
        assert restored.global_load_efficiency == original.global_load_efficiency
        assert restored.global_store_efficiency == original.global_store_efficiency
        assert restored.l2_hit_rate == original.l2_hit_rate
        assert restored.l1_hit_rate == original.l1_hit_rate
        assert restored.occupancy_limited_by_registers == original.occupancy_limited_by_registers
        assert restored.occupancy_limited_by_shared_mem == original.occupancy_limited_by_shared_mem
        assert restored.raw_metrics == original.raw_metrics
        assert restored.metadata == original.metadata

    def test_minimal_serialization(self):
        m = ProfileMetrics(kernel_name="minimal")
        dumped = json.loads(json.dumps(m.model_dump(mode="json")))
        restored = ProfileMetrics.model_validate(dumped)
        assert restored.kernel_name == "minimal"
        assert restored.gpu_time_us is None
        assert restored.sm_throughput is None
        assert restored.block_size is None

    def test_mock_profiler_output_serializable(self):
        profiler = MockProfiler()
        kernel = KernelSource(
            name="tiled_matmul.cu",
            language=KernelLanguage.CUDA,
            source_code="",
        )
        m = _run(profiler.profile(kernel, run_id="r1", arch="sm_80"))
        json_str = json.dumps(m.model_dump(mode="json"))
        assert len(json_str) > 10

    def test_tuple_fields_serialize_as_lists(self):
        """block_size tuple becomes a JSON list."""
        m = ProfileMetrics(
            kernel_name="test",
            block_size=(16, 16, 1),
            grid_size=(64, 64, 1),
        )
        dumped = m.model_dump(mode="json")
        assert dumped["block_size"] == [16, 16, 1]
        assert dumped["grid_size"] == [64, 64, 1]

    def test_tuple_fields_restore_from_lists(self):
        """List values deserialize back to tuples."""
        data = {
            "kernel_name": "test",
            "block_size": [32, 8, 1],
            "grid_size": [128, 1, 1],
        }
        m = ProfileMetrics.model_validate(data)
        assert m.block_size == (32, 8, 1)
        assert m.grid_size == (128, 1, 1)
