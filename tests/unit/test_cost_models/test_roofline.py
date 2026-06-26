
"""Tests for the roofline cost model."""

from cuda_opt_agent.cost_models.analytical.roofline import RooflineModel
from cuda_opt_agent.schemas.cost_model import BoundType
from cuda_opt_agent.schemas.features import KernelFeatureVector


class TestRooflineModel:

    def test_memory_bound(self):
        model = RooflineModel()
        features = KernelFeatureVector(
            sm_throughput_pct=30.0, memory_throughput_pct=85.0
        )
        est = model.predict(features, metrics=None)
        assert est.bound_type == BoundType.MEMORY
        assert est.confidence > 0.5

    def test_compute_bound(self):
        model = RooflineModel()
        features = KernelFeatureVector(
            sm_throughput_pct=90.0, memory_throughput_pct=20.0
        )
        est = model.predict(features, metrics=None)
        assert est.bound_type == BoundType.COMPUTE

    def test_mixed(self):
        model = RooflineModel()
        features = KernelFeatureVector(
            sm_throughput_pct=60.0, memory_throughput_pct=55.0
        )
        est = model.predict(features, metrics=None)
        assert est.bound_type == BoundType.MIXED

    def test_no_data_low_confidence(self):
        model = RooflineModel()
        est = model.predict(KernelFeatureVector(), metrics=None)
        assert est.confidence <= 0.2

    def test_speedup_with_improvements(self):
        model = RooflineModel()
        base = KernelFeatureVector(
            ilp_estimate=1.0, is_coalesced=False, achieved_occupancy=0.4,
            bank_conflict_count=4,
        )
        variant = KernelFeatureVector(
            ilp_estimate=2.0, is_coalesced=True, achieved_occupancy=0.7,
            bank_conflict_count=0,
        )
        pred = model.predict_speedup(base, variant)
        assert pred.predicted_speedup > 1.0
        assert len(pred.improvement_areas) > 0

    def test_speedup_no_change(self):
        model = RooflineModel()
        f = KernelFeatureVector()
        pred = model.predict_speedup(f, f)
        assert pred.predicted_speedup == 1.0
