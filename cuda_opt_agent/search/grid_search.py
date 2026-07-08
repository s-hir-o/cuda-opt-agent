"""Grid search — exhaustive enumeration of parameter combinations.

Generates the Cartesian product of all parameter values from each
transform's parameter_schema. Produces one batch per call to
generate_candidates() with configurable batch_size.
"""

from __future__ import annotations

import itertools

from cuda_opt_agent.search.base import (
    SearchContext,
    SearchStrategy,
    extract_param_grid,
)
from cuda_opt_agent.schemas.transform import TransformSpec
from cuda_opt_agent.transforms.base import Transformation


class GridSearch(SearchStrategy):
    """Exhaustive grid search over transform parameter space.

    On first call to generate_candidates(), enumerates all parameter
    combinations for all applicable transforms. Returns them in batches
    of batch_size until exhausted.
    """

    def __init__(self, batch_size: int = 8) -> None:
        self._batch_size = max(1, batch_size)
        self._queue: list[TransformSpec] = []
        self._initialized = False
        self._total_candidates = 0

    @property
    def name(self) -> str:
        return "grid"

    def generate_candidates(
        self, context: SearchContext
    ) -> list[TransformSpec]:
        """Generate candidates. On first call, enumerates full grid.

        Args:
            context: Search context with available transforms.

        Returns:
            Next batch of TransformSpec objects, or [] if exhausted.
        """
        if not self._initialized:
            self._queue = self._enumerate_all(context.transforms)
            self._total_candidates = len(self._queue)
            self._initialized = True

        if not self._queue:
            return []

        batch = self._queue[: self._batch_size]
        self._queue = self._queue[self._batch_size :]
        return batch

    def reset(self) -> None:
        self._queue = []
        self._initialized = False
        self._total_candidates = 0

    @property
    def total_candidates(self) -> int:
        """Total number of candidates in the full grid (after enumeration)."""
        return self._total_candidates

    @staticmethod
    def _enumerate_all(
        transforms: list[Transformation],
    ) -> list[TransformSpec]:
        """Enumerate all parameter combinations for all transforms.

        For each transform, parses its parameter_schema to extract
        candidate values, then generates the Cartesian product.
        """
        all_specs: list[TransformSpec] = []

        for transform in transforms:
            schema = transform.parameter_schema
            grid = extract_param_grid(schema)

            if not grid:
                # No parameters — just the transform with defaults
                all_specs.append(TransformSpec(
                    transform_name=transform.name,
                    params={},
                    reason="grid: no params",
                ))
                continue

            param_names = sorted(grid.keys())
            value_combos = itertools.product(*(grid[n] for n in param_names))

            for combo in value_combos:
                params = dict(zip(param_names, combo))
                all_specs.append(TransformSpec(
                    transform_name=transform.name,
                    params=params,
                    reason=f"grid: {params}",
                ))

        return all_specs
