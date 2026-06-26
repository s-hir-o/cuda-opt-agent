"""Stub for future ML cost model (XGBoost/NN). Import-safe — raises on use."""

from __future__ import annotations

from cuda_opt_agent.cost_models.base import CostModel
from cuda_opt_agent.schemas.cost_model import CostEstimate, ModelPrediction
from cuda_opt_agent.schemas.features import KernelFeatureVector
from cuda_opt_agent.schemas.metrics import ProfileMetrics


class LearnedCostModel(CostModel):

    @property
    def name(self) -> str:
        return "learned"

    def predict(self, features: KernelFeatureVector, metrics: ProfileMetrics | None) -> CostEstimate:
        raise NotImplementedError("Learned cost model not yet implemented.")

    def predict_speedup(self, baseline: KernelFeatureVector, variant: KernelFeatureVector) -> ModelPrediction:
        raise NotImplementedError("Learned cost model not yet implemented.")
