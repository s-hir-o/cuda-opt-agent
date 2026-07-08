"""Stub for future evolutionary search strategy.

Would use genetic algorithms to evolve parameter combinations
over generations. Import-safe — raises NotImplementedError on use.
"""

from __future__ import annotations

from cuda_opt_agent.search.base import SearchContext, SearchResult, SearchStrategy
from cuda_opt_agent.schemas.transform import TransformSpec


class EvolutionarySearch(SearchStrategy):

    def __init__(
        self,
        population_size: int = 20,
        generations: int = 10,
        mutation_rate: float = 0.1,
        seed: int = 42,
    ) -> None:
        self._pop_size = population_size
        self._generations = generations
        self._mutation_rate = mutation_rate

    @property
    def name(self) -> str:
        return "evolutionary"

    @property
    def is_adaptive(self) -> bool:
        return True

    def generate_candidates(self, context: SearchContext) -> list[TransformSpec]:
        raise NotImplementedError(
            "Evolutionary search not yet implemented. "
            "Requires DEAP or custom GA framework."
        )

    def report_result(self, result: SearchResult) -> None:
        raise NotImplementedError

    def reset(self) -> None:
        pass
