"""Tests for the transform registry."""

import pytest

from cuda_opt_agent.core.exceptions import ConfigurationError
from cuda_opt_agent.schemas.bottleneck import BottleneckClass
from cuda_opt_agent.transforms.registry import (
    get_transform,
    list_transforms,
    transforms_for_bottleneck,
)

EXPECTED = [
    "shared_memory_tiling",
    "loop_unrolling",
    "vectorized_loads",
    "register_blocking",
    "thread_coarsening",
    "warp_shuffle",
    "kernel_fusion",
]


class TestRegistry:

    def test_all_registered(self):
        names = list_transforms()
        for t in EXPECTED:
            assert t in names

    def test_exact_count(self):
        assert len(list_transforms()) == len(EXPECTED)

    def test_get_transform(self):
        for t in EXPECTED:
            assert get_transform(t).name == t

    def test_unknown_raises(self):
        with pytest.raises(ConfigurationError):
            get_transform("does_not_exist_xyz")

    def test_deterministic_order(self):
        a = list_transforms()
        b = list_transforms()
        assert a == b

    def test_bottleneck_memory(self):
        names = [t.name for t in transforms_for_bottleneck(BottleneckClass.MEMORY_BOUND)]
        assert "shared_memory_tiling" in names
        assert "vectorized_loads" in names
        assert "thread_coarsening" in names

    def test_bottleneck_compute(self):
        names = [t.name for t in transforms_for_bottleneck(BottleneckClass.COMPUTE_BOUND)]
        assert "loop_unrolling" in names
        assert "register_blocking" in names

    def test_schema_is_object(self):
        for t in EXPECTED:
            s = get_transform(t).parameter_schema
            assert isinstance(s, dict)
            assert s.get("type") == "object"