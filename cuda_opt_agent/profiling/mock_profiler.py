




"""Mock profiler for testing without a GPU."""

from __future__ import annotations

from cuda_opt_agent.profiling.profiler import Profiler
from cuda_opt_agent.schemas.kernel import KernelSource, KernelVariant
from cuda_opt_agent.schemas.metrics import ProfileMetrics


class MockProfiler(Profiler):
    """Returns deterministic metrics. Configurable per kernel name."""

    _PRESETS: dict[str, dict] = {
        "naive_matmul": {
            "gpu_time_us": 850.0,
            "sm_throughput": 35.0,
            "dram_throughput": 78.0,
            "achieved_occupancy": 0.50,
            "registers_per_thread": 40,
            "global_load_efficiency": 25.0,
            "global_store_efficiency": 100.0,
            "occupancy_limited_by_registers": True,
        },
        "tiled_matmul": {
            "gpu_time_us": 120.0,
            "sm_throughput": 65.0,
            "dram_throughput": 45.0,
            "achieved_occupancy": 0.75,
            "registers_per_thread": 32,
            "global_load_efficiency": 90.0,
            "global_store_efficiency": 100.0,
            "occupancy_limited_by_registers": False,
        },
        "compute_bound_kernel": {
            "gpu_time_us": 50.0,
            "sm_throughput": 92.0,
            "dram_throughput": 15.0,
            "achieved_occupancy": 0.85,
            "registers_per_thread": 64,
            "global_load_efficiency": 95.0,
            "global_store_efficiency": 95.0,
            "occupancy_limited_by_registers": False,
            "occupancy_limited_by_shared_mem": False,
        },
        "memory_bound_kernel": {
            "gpu_time_us": 300.0,
            "sm_throughput": 20.0,
            "dram_throughput": 90.0,
            "achieved_occupancy": 0.40,
            "registers_per_thread": 48,
            "global_load_efficiency": 30.0,
            "global_store_efficiency": 30.0,
            "occupancy_limited_by_registers": True,
            "occupancy_limited_by_shared_mem": False,
        },
    }

    def __init__(self, arch: str = "sm_80", gpu_name: str = "NVIDIA A100") -> None:
        self._arch = arch
        self._gpu_name = gpu_name

    async def is_available(self) -> bool:
        return True

    async def profile(
        self,
        kernel: KernelSource | KernelVariant,
        run_id: str,
        arch: str,
    ) -> ProfileMetrics:
        name = kernel.name if isinstance(kernel, KernelSource) else kernel.variant_id
        short = name.rsplit(".", 1)[0]

        # Try exact match, then prefix match (for variants), then fallback
        if short in self._PRESETS:
            kernel_name = short
            preset = self._PRESETS[short]
        else:
            matched_key = None
            for key in self._PRESETS:
                if short.startswith(key):
                    matched_key = key
                    break
            if matched_key:
                kernel_name = matched_key
                preset = self._PRESETS[matched_key]
            else:
                kernel_name = short
                preset = self._PRESETS["naive_matmul"]

        return ProfileMetrics(
            kernel_name=kernel_name,
            gpu_name=self._gpu_name,
            arch=arch or self._arch,
            block_size=(16, 16, 1),
            grid_size=(64, 64, 1),
            metadata={"variant_id": getattr(kernel, "variant_id", None)},
            **{k: v for k, v in preset.items() if k in ProfileMetrics.model_fields},
        )