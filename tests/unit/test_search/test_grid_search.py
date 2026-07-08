"""Tests for GridSearch strategy."""

from cuda_opt_agent.search.grid_search import GridSearch
from cuda_opt_agent.search.base import SearchContext, SearchResult
from cuda_opt_agent.schemas.bottleneck import BottleneckClass, BottleneckResult
from cuda_opt_agent.schemas.kernel import KernelLanguage, KernelSource
from cuda_opt_agent.schemas.transform import TransformSpec
from cuda_opt_agent.transforms.text.loop_unrolling import LoopUnrolling
from cuda_opt_agent.transforms.text.shared_memory_tiling import SharedMemoryTiling

KERNEL = KernelSource(name="test.cu", language=KernelLanguage.CUDA, source_code="")
BOTTLENECK = BottleneckResult(primary=BottleneckClass.MEMORY_BOUND, confidence=0.9)


class TestGridSearch:

    def test_name(self):
        gs = GridSearch()
        assert gs.name == "grid"

    def test_not_adaptive(self):
        gs = GridSearch()
        assert gs.is_adaptive is False

    def test_single_transform_no_enum(self):
        """Transform with only a default param generates one candidate."""
        gs = GridSearch(batch_size=100)

        # SharedMemoryTiling has tile_size with default=16 but also minimum=1
        # Without max, extract_param_grid will use default only
        transforms = [SharedMemoryTiling()]
        context = SearchContext(transforms=transforms)

        candidates = gs.generate_candidates(context)
        assert len(candidates) >= 1
        assert all(c.transform_name == "shared_memory_tiling" for c in candidates)

    def test_transform_with_enum(self):
        """Transform with enum params generates all enum values."""
        # VectorizedLoads has vector_width enum: [2, 4]
        from cuda_opt_agent.transforms.text.vectorized_loads import VectorizedLoads

        gs = GridSearch(batch_size=100)
        transforms = [VectorizedLoads()]
        context = SearchContext(transforms=transforms)

        candidates = gs.generate_candidates(context)
        widths = {c.params.get("vector_width") for c in candidates}
        assert 2 in widths
        assert 4 in widths

    def test_batching(self):
        """Large grid returned in batches."""
        from cuda_opt_agent.transforms.text.vectorized_loads import VectorizedLoads

        gs = GridSearch(batch_size=1)
        transforms = [VectorizedLoads()]
        context = SearchContext(transforms=transforms)

        batch1 = gs.generate_candidates(context)
        assert len(batch1) == 1

        batch2 = gs.generate_candidates(context)
        assert len(batch2) >= 1

        # After all batches, returns empty
        while gs.generate_candidates(context):
            pass
        assert gs.generate_candidates(context) == []

    def test_multiple_transforms(self):
        """Grid includes candidates from all transforms."""
        from cuda_opt_agent.transforms.text.warp_shuffle import WarpShuffleReduction

        gs = GridSearch(batch_size=100)
        transforms = [SharedMemoryTiling(), WarpShuffleReduction()]
        context = SearchContext(transforms=transforms)

        candidates = gs.generate_candidates(context)
        names = {c.transform_name for c in candidates}
        assert "shared_memory_tiling" in names
        assert "warp_shuffle" in names

    def test_empty_transforms(self):
        gs = GridSearch()
        context = SearchContext(transforms=[])
        assert gs.generate_candidates(context) == []

    def test_reset(self):
        gs = GridSearch(batch_size=1)
        transforms = [SharedMemoryTiling()]
        context = SearchContext(transforms=transforms)

        # Exhaust all candidates
        while gs.generate_candidates(context):
            pass
        assert gs.generate_candidates(context) == []

        # Reset should allow regeneration
        gs.reset()
        candidates = gs.generate_candidates(context)
        assert len(candidates) >= 1

    def test_result_reported_noop(self):
        """GridSearch ignores report_result — no error."""
        gs = GridSearch()
        gs.report_result(SearchResult(
            spec=TransformSpec(transform_name="test", params={}),
            speedup=2.0,
        ))
        # No error raised — grid search is non-adaptive
