
"""Tests for the occupancy cost model."""

from cuda_opt_agent.cost_models.analytical.occupancy_model import OccupancyModel
from cuda_opt_agent.schemas.features import KernelFeatureVector


class TestOccupancyModel:

    def test_high_occupancy(self):
        model = OccupancyModel(arch="sm_80")
        features = KernelFeatureVector(
            registers_per_thread=32,
            block_size_x=256,
            block_size_y=1,
        )
        est = model.predict(features, metrics=None)
        # 256 threads * 32 regs = 8192 regs/block, 65536/8192 = 8 blocks
        # 8 * 256 = 2048 threads = full occupancy
        assert est.predicted_occupancy is not None
        assert est.predicted_occupancy >= 0.9

    def test_register_limited(self):
        model = OccupancyModel(arch="sm_80")
        features = KernelFeatureVector(
            registers_per_thread=128,
            block_size_x=256,
        )
        est = model.predict(features, metrics=None)
        assert est.predicted_occupancy is not None
        assert est.predicted_occupancy < 0.5

    def test_name(self):
        model = OccupancyModel()
        assert model.name == "occupancy"
        assert model.is_learned is False

    def test_speedup_prediction(self):
        model = OccupancyModel()
        base = KernelFeatureVector(registers_per_thread=64, block_size_x=256)
        variant = KernelFeatureVector(registers_per_thread=32, block_size_x=256)
        pred = model.predict_speedup(base, variant)
        assert pred.predicted_speedup >= 1.0


