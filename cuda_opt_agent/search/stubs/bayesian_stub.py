"""Stub for future Bayesian optimization search strategy.

Would use a surrogate model (Gaussian Process or TPE) to model
the objective function and select informative candidates.
Import-safe — raises NotImplementedError on use.
"""

from __future__ import annotations

from cuda_opt_agent.search.base import SearchContext, SearchResult, SearchStrategy
from cuda_opt_agent.schemas.transform import TransformSpec


class BayesianSearch(SearchStrategy):

    def __init__(self, max_samples: int = 50, seed: int = 42) -> None:
        self._max_samples = max_samples

    @property
    def name(self) -> str:
        return "bayesian"

    @property
    def is_adaptive(self) -> bool:
        return True

    def generate_candidates(self, context: SearchContext) -> list[TransformSpec]:
        raise NotImplementedError(
            "Bayesian optimization not yet implemented. "
            "Requires scikit-optimize or optuna dependency."
        )

    def report_result(self, result: SearchResult) -> None:
        raise NotImplementedError

    def reset(self) -> None:
        pass
