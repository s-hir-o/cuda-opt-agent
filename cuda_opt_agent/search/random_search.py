"""Random search — uniform random sampling from parameter space.

Samples random parameter combinations for each transform.
Supports reproducible results via seed. Generates one batch per call
to generate_candidates().
"""

from __future__ import annotations

import random as _random
from typing import Any

from cuda_opt_agent.search.base import (
    SearchContext,
    SearchStrategy,
    extract_param_grid,
)
from cuda_opt_agent.schemas.transform import TransformSpec
from cuda_opt_agent.transforms.base import Transformation


class RandomSearch(SearchStrategy):
    """Random sampling of transform parameter space.

    Generates up to max_samples total candidates across all calls
    to generate_candidates(). Samples uniformly from each parameter's
    value set (extracted from JSON Schema).
    """

    def __init__(
        self,
        max_samples: int = 50,
        seed: int = 42,
        batch_size: int = 8,
    ) -> None:
        self._max_samples = max(1, max_samples)
        self._batch_size = max(1, batch_size)
        self._seed = seed
        self._rng = _random.Random(seed)
        self._generated = 0
        self._transform_cache: list[tuple[Transformation, dict[str, list[Any]]]] = []

    @property
    def name(self) -> str:
        return "random"

    @property
    def is_adaptive(self) -> bool:
        return False

    def generate_candidates(
        self, context: SearchContext
    ) -> list[TransformSpec]:
        """Generate a random batch of candidates.

        Args:
            context: Search context with available transforms.

        Returns:
            Batch of TransformSpec objects, or [] if max_samples reached.
        """
        remaining = self._max_samples - self._generated
        if remaining <= 0:
            return []

        # Cache transform grids on first call
        if not self._transform_cache:
            self._transform_cache = [
                (t, extract_param_grid(t.parameter_schema))
                for t in context.transforms
            ]

        if not self._transform_cache:
            return []

        batch_size = min(self._batch_size, remaining)
        batch: list[TransformSpec] = []

        for _ in range(batch_size):
            spec = self._sample_one()
            if spec is not None:
                batch.append(spec)

        self._generated += len(batch)
        return batch

    def reset(self) -> None:
        self._generated = 0
        self._rng = _random.Random(self._seed)
        self._transform_cache = []

    @property
    def samples_remaining(self) -> int:
        return max(0, self._max_samples - self._generated)

    def _sample_one(self) -> TransformSpec | None:
        """Sample a single random TransformSpec."""
        if not self._transform_cache:
            return None

        # Pick a random transform
        transform, grid = self._rng.choice(self._transform_cache)

        if not grid:
            return TransformSpec(
                transform_name=transform.name,
                params={},
                reason="random: no params",
            )

        # Sample random values for each parameter
        params: dict[str, Any] = {}
        for param_name, values in grid.items():
            params[param_name] = self._rng.choice(values)

        return TransformSpec(
            transform_name=transform.name,
            params=params,
            reason=f"random: {params}",
        )
