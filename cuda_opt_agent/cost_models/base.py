"""Abstract cost model interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

from cuda_opt_agent.schemas.cost_model import CostEstimate, ModelPrediction
from cuda_opt_agent.schemas.features import KernelFeatureVector
from cuda_opt_agent.schemas.metrics import ProfileMetrics


class CostModel(ABC):
    """Predict kernel performance from features."""

    @abstractmethod
    def predict(
        self,
        features: KernelFeatureVector,
        metrics: ProfileMetrics | None,
    ) -> CostEstimate:
        """Predict performance. Never raises — returns low-confidence on failure."""

    @abstractmethod
    def predict_speedup(
        self,
        baseline_features: KernelFeatureVector,
        variant_features: KernelFeatureVector,
    ) -> ModelPrediction:
        """Predict relative speedup of variant vs baseline."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Model identifier."""

    @property
    def is_learned(self) -> bool:
        """False for analytical, True for ML."""
        return False
