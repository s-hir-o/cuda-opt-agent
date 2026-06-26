
"""Tests for CUDA source feature extraction."""

from cuda_opt_agent.features.extractors.cuda_source import CudaSourceExtractor
from cuda_opt_agent.schemas.kernel import KernelLanguage, KernelSource


class TestCudaSourceExtractor:

    def _extract(self, source: str):
        ext = CudaSourceExtractor()
        kernel = KernelSource(name="test.cu", language=KernelLanguage.CUDA, source_code=source)
        return ext.extract(kernel, metrics=None, analysis=None)

    def test_shared_memory_detected(self):
        v = self._extract("__shared__ float tile[16][16];\n__global__ void k() {}")
        assert v.has_shared_memory is True

    def test_no_shared_memory(self):
        v = self._extract("__global__ void k() { float x = 1.0f; }")
        assert v.has_shared_memory is False

    def test_loop_depth(self):
        src = "__global__ void k() {\n  for(int i=0;i<10;i++) {\n    for(int j=0;j<10;j++) {\n    }\n  }\n}"
        v = self._extract(src)
        assert v.loop_nest_depth == 2

    def test_name(self):
        ext = CudaSourceExtractor()
        assert ext.name == "cuda_source"
        assert "source" in ext.required_inputs
