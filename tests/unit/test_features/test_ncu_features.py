
"""Tests for NCU feature extraction."""

from cuda_opt_agent.features.extractors.ncu_features import NCUFeatureExtractor
from cuda_opt_agent.schemas.kernel import KernelLanguage, KernelSource
from cuda_opt_agent.schemas.metrics import ProfileMetrics


class TestNCUFeatureExtractor:

    def _extract(self, metrics: ProfileMetrics):
        ext = NCUFeatureExtractor()
        kernel = KernelSource(name="test.cu", language=KernelLanguage.CUDA, source_code="")
        return ext.extract(kernel, metrics=metrics, analysis=None)

    def test_throughput(self):
        m = ProfileMetrics(kernel_name="test", sm_throughput=80.0, dram_throughput=45.0)
        v = self._extract(m)
        assert v.sm_throughput_pct == 80.0
        assert v.memory_throughput_pct == 45.0

    def test_occupancy(self):
        m = ProfileMetrics(
            kernel_name="test",
            achieved_occupancy=0.75,
            registers_per_thread=40,
            occupancy_limited_by_registers=True,
        )
        v = self._extract(m)
        assert v.achieved_occupancy == 0.75
        assert v.registers_per_thread == 40
        assert v.occupancy_limited_by_registers is True

    def test_block_grid_size(self):
        m = ProfileMetrics(
            kernel_name="test",
            block_size=(16, 16, 1),
            grid_size=(64, 64, 1),
        )
        v = self._extract(m)
        assert v.block_size_x == 16
        assert v.block_size_y == 16
        assert v.grid_size_x == 64

    def test_coalesced_from_efficiency(self):
        m = ProfileMetrics(kernel_name="test", global_load_efficiency=95.0)
        v = self._extract(m)
        assert v.is_coalesced is True

    def test_uncoalesced_from_efficiency(self):
        m = ProfileMetrics(kernel_name="test", global_load_efficiency=25.0)
        v = self._extract(m)
        assert v.is_coalesced is False

    def test_none_metrics_returns_empty(self):
        ext = NCUFeatureExtractor()
        kernel = KernelSource(name="test.cu", language=KernelLanguage.CUDA, source_code="")
        v = ext.extract(kernel, metrics=None, analysis=None)
        assert v.sm_throughput_pct is None
