"""Feature pipeline — runs all extractors and merges results."""

from __future__ import annotations

from cuda_opt_agent.features.base import FeatureExtractor
from cuda_opt_agent.schemas.analysis import KernelAnalysis
from cuda_opt_agent.schemas.features import KernelFeatureVector
from cuda_opt_agent.schemas.kernel import KernelSource, KernelVariant
from cuda_opt_agent.schemas.metrics import ProfileMetrics


class FeaturePipeline:
    """Compose multiple FeatureExtractors into a single extraction call.

    Each extractor populates fields it can; later extractors fill remaining gaps.
    """

    def __init__(self, extractors: list[FeatureExtractor]) -> None:
        self._extractors = extractors

    def extract(
        self,
        kernel: KernelSource | KernelVariant,
        metrics: ProfileMetrics | None = None,
        analysis: KernelAnalysis | None = None,
    ) -> KernelFeatureVector:
        """Run all extractors and merge into a single feature vector."""
        result = KernelFeatureVector()
        for extractor in self._extractors:
            partial = extractor.extract(kernel, metrics, analysis)
            result = _merge(result, partial)
        return result


def _merge(base: KernelFeatureVector, overlay: KernelFeatureVector) -> KernelFeatureVector:
    """Merge two feature vectors: overlay's non-None values take priority."""
    base_data = base.model_dump()
    overlay_data = overlay.model_dump()
    for key, value in overlay_data.items():
        if value is not None:
            base_data[key] = value
    return KernelFeatureVector(**base_data)
