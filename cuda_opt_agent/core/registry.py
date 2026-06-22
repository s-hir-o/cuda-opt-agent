"""Plugin registry for transforms, cost models, and analyzers."""

from __future__ import annotations

from typing import Any, TypeVar

from cuda_opt_agent.core.exceptions import ConfigurationError

T = TypeVar("T")


class PluginRegistry:
    """Generic plugin registry with autodiscovery support.

    Stores named plugins keyed by string name. Plugins are registered
    via decorator or manual register() call. Supports type-safe retrieval.
    """

    def __init__(self, kind: str) -> None:
        self._kind = kind
        self._plugins: dict[str, type] = {}
        self._instances: dict[str, Any] = {}

    def register(self, name: str | None = None):
        """Class decorator to register a plugin.

        Args:
            name: Registry name. Defaults to class name (lowercased).
        """

        def decorator(cls: type[T]) -> type[T]:
            key = (name or cls.__name__).lower()
            if key in self._plugins:
                raise ConfigurationError(
                    f"{self._kind} plugin '{key}' already registered "
                    f"(existing: {self._plugins[key].__name__}, new: {cls.__name__})"
                )
            self._plugins[key] = cls
            return cls

        return decorator

    def register_instance(self, name: str, instance: Any) -> None:
        """Register a pre-constructed instance."""
        key = name.lower()
        self._plugins[key] = type(instance)
        self._instances[key] = instance

    def get_class(self, name: str) -> type:
        """Get registered plugin class by name."""
        key = name.lower()
        if key not in self._plugins:
            available = ", ".join(sorted(self._plugins.keys())) or "(none)"
            raise ConfigurationError(
                f"{self._kind} plugin '{name}' not found. Available: {available}"
            )
        return self._plugins[key]

    def get(self, name: str) -> Any:
        """Get or create plugin instance by name (singleton per name)."""
        key = name.lower()
        if key not in self._instances:
            cls = self.get_class(name)
            self._instances[key] = cls()
        return self._instances[key]

    def list_plugins(self) -> list[str]:
        """Return sorted list of registered plugin names."""
        return sorted(self._plugins.keys())

    def clear(self) -> None:
        """Remove all registrations. Primarily for testing."""
        self._plugins.clear()
        self._instances.clear()


# ── Global registries (initialized at import, populated by modules) ──

transform_registry = PluginRegistry("transform")
cost_model_registry = PluginRegistry("cost_model")
analyzer_registry = PluginRegistry("analyzer")
feature_extractor_registry = PluginRegistry("feature_extractor")
search_strategy_registry = PluginRegistry("search_strategy")
