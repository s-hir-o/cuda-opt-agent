"""Tests for JSON Schema parameter grid extraction."""

from cuda_opt_agent.search.base import extract_param_grid


class TestExtractParamGrid:

    def test_enum_values(self):
        schema = {
            "type": "object",
            "properties": {
                "op": {"type": "string", "enum": ["sum", "max", "min"]},
            },
        }
        grid = extract_param_grid(schema)
        assert grid["op"] == ["sum", "max", "min"]

    def test_integer_with_min_max(self):
        schema = {
            "type": "object",
            "properties": {
                "factor": {"type": "integer", "minimum": 1, "maximum": 8},
            },
        }
        grid = extract_param_grid(schema)
        assert 1 in grid["factor"]
        assert 8 in grid["factor"]
        assert all(isinstance(v, int) for v in grid["factor"])

    def test_default_only(self):
        schema = {
            "type": "object",
            "properties": {
                "tile_size": {"type": "integer", "default": 16},
            },
        }
        grid = extract_param_grid(schema)
        assert grid["tile_size"] == [16]

    def test_no_properties(self):
        schema = {"type": "object", "properties": {}}
        grid = extract_param_grid(schema)
        assert grid == {}

    def test_empty_schema(self):
        grid = extract_param_grid({})
        assert grid == {}

    def test_multiple_params(self):
        schema = {
            "type": "object",
            "properties": {
                "block_m": {"type": "integer", "minimum": 1, "maximum": 4},
                "block_n": {"type": "integer", "minimum": 1, "maximum": 4},
            },
        }
        grid = extract_param_grid(schema)
        assert "block_m" in grid
        assert "block_n" in grid
        assert len(grid["block_m"]) >= 2
        assert len(grid["block_n"]) >= 2

    def test_real_transform_schemas(self):
        """Verify all registered transforms have parseable schemas."""
        from cuda_opt_agent.transforms.text.shared_memory_tiling import SharedMemoryTiling
        from cuda_opt_agent.transforms.text.loop_unrolling import LoopUnrolling
        from cuda_opt_agent.transforms.text.vectorized_loads import VectorizedLoads
        from cuda_opt_agent.transforms.text.register_blocking import RegisterBlocking
        from cuda_opt_agent.transforms.text.thread_coarsening import ThreadCoarsening
        from cuda_opt_agent.transforms.text.warp_shuffle import WarpShuffleReduction

        transforms = [
            SharedMemoryTiling(), LoopUnrolling(), VectorizedLoads(),
            RegisterBlocking(), ThreadCoarsening(), WarpShuffleReduction(),
        ]

        for t in transforms:
            grid = extract_param_grid(t.parameter_schema)
            assert isinstance(grid, dict), f"{t.name} schema not parseable"
            # Each transform should have at least one parseable parameter
            # (except kernel_fusion which has optional params)

    def test_int_grid_capped(self):
        """Grid should not generate more than 8 values."""
        schema = {
            "type": "object",
            "properties": {
                "x": {"type": "integer", "minimum": 1, "maximum": 1000},
            },
        }
        grid = extract_param_grid(schema)
        assert len(grid["x"]) <= 8

    def test_float_grid(self):
        schema = {
            "type": "object",
            "properties": {
                "rate": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            },
        }
        grid = extract_param_grid(schema)
        assert len(grid["rate"]) == 5
        assert grid["rate"][0] == 0.0
        assert grid["rate"][-1] == 1.0
