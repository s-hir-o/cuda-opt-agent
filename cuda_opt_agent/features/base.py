"""Abstract feature extractor."""

from __future__ import annotations

from abc import ABC, abstractmethod

from cuda_opt_agent.schemas.analysis import KernelAnalysis
from cuda_opt_agent.schemas.features import KernelFeatureVector
from cuda_opt_agent.schemas.kernel import KernelSource, KernelVariant
from cuda_opt_agent.schemas.metrics import ProfileMetrics


class FeatureExtractor(ABC):
    """Extract features from available kernel data into a KernelFeatureVector."""

    @abstractmethod
    def extract(
        self,
        kernel: KernelSource | KernelVariant,
        metrics: ProfileMetrics | None,
        analysis: KernelAnalysis | None,
    ) -> KernelFeatureVector:
        """Extract features. Absent data → None fields, never raises."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Extractor identifier."""

    @property
    @abstractmethod
    def required_inputs(self) -> list[str]:
        """Which inputs are needed: 'source', 'metrics', 'analysis'."""
