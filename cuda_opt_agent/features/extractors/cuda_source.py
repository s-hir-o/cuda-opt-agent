
"""Extract features from CUDA source code using regex heuristics."""

from __future__ import annotations

import re

from cuda_opt_agent.features.base import FeatureExtractor
from cuda_opt_agent.schemas.analysis import KernelAnalysis
from cuda_opt_agent.schemas.features import KernelFeatureVector
from cuda_opt_agent.schemas.kernel import KernelSource, KernelVariant
from cuda_opt_agent.schemas.metrics import ProfileMetrics


class CudaSourceExtractor(FeatureExtractor):

    @property
    def name(self) -> str:
        return "cuda_source"

    @property
    def required_inputs(self) -> list[str]:
        return ["source"]

    def extract(
        self,
        kernel: KernelSource | KernelVariant,
        metrics: ProfileMetrics | None,
        analysis: KernelAnalysis | None,
    ) -> KernelFeatureVector:
        src = kernel.source_code

        # Count nested loop depth
        loop_depth = _max_loop_depth(src)

        return KernelFeatureVector(
            has_shared_memory=bool(re.search(r"__shared__", src)),
            loop_nest_depth=loop_depth,
            block_size_x=_extract_const(src, r"blockDim\.x\s*=\s*(\d+)"),
            block_size_y=_extract_const(src, r"blockDim\.y\s*=\s*(\d+)"),
            block_size_z=_extract_const(src, r"blockDim\.z\s*=\s*(\d+)"),
            grid_size_x=_extract_const(src, r"gridDim\.x\s*=\s*(\d+)"),
            grid_size_y=_extract_const(src, r"gridDim\.y\s*=\s*(\d+)"),
        )


def _max_loop_depth(source: str) -> int:
    """Estimate maximum loop nesting depth from braces."""
    max_depth = 0
    current = 0
    for line in source.splitlines():
        stripped = line.strip()
        if re.match(r"(for|while|do)\s*[\({]", stripped):
            current += 1
            max_depth = max(max_depth, current)
        if "}" in stripped:
            current = max(0, current - 1)
    return max_depth


def _extract_const(source: str, pattern: str) -> int | None:
    """Extract a single integer constant matching a regex pattern."""
    m = re.search(pattern, source)
    if m:
        try:
            return int(m.group(1))
        except (ValueError, IndexError):
            return None
    return None
