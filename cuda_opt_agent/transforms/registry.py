"""Transform registry — autodiscovery + lookup by name or bottleneck class.

Wraps the global transform_registry from core/registry.py with
convenience methods for the coordinator agent.
"""

from __future__ import annotations

from typing import Any

from cuda_opt_agent.core.exceptions import ConfigurationError
from cuda_opt_agent.core.registry import transform_registry
from cuda_opt_agent.schemas.bottleneck import BottleneckClass
from cuda_opt_agent.transforms.base import Transformation


def get_transform(name: str) -> Transformation:
    """Get a transform instance by name. Raises ConfigurationError if not found."""
    try:
        return transform_registry.get(name)
    except KeyError:
        raise ConfigurationError(f"Transform '{name}' is not registered")


def list_transforms() -> list[str]:
    """Return all registered transform names."""
    return sorted(transform_registry.list_plugins())


def transforms_for_bottleneck(bottleneck: BottleneckClass) -> list[Transformation]:
    """Return all transforms that target the given bottleneck class.

    Results are sorted by transform name for deterministic behaviour.
    """
    result: list[Transformation] = []
    for name in sorted(transform_registry.list_plugins()):
        t = transform_registry.get(name)
        if bottleneck in t.targets:
            result.append(t)
    return result


def register_transform(cls: type[Transformation]) -> type[Transformation]:
    """Decorator to register a transform class under its .name property."""
    instance = cls()
    transform_registry.register(name=instance.name)(cls)
    return cls
    