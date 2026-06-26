"""Batch feature vector operations."""

from __future__ import annotations

import numpy as np

from cuda_opt_agent.schemas.features import KernelFeatureVector
from cuda_opt_agent.features.normalizer import fill_defaults


def batch_to_numpy(vectors: list[KernelFeatureVector]) -> np.ndarray:
    """Convert a list of feature vectors to a 2D numpy array (n_samples, n_features).

    None values are filled with 0.0 before stacking.
    """
    if not vectors:
        return np.array([], dtype=np.float32).reshape(0, 0)
    filled = [fill_defaults(v) for v in vectors]
    return np.stack([v.to_numpy(fill=0.0) for v in filled])


def feature_names() -> list[str]:
    """Return column names matching the numpy array columns."""
    return KernelFeatureVector().field_names()


LAYER 3.7 — Cost Models
