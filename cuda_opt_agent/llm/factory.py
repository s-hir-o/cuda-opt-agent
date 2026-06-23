"""Factory: creates the correct LLMProvider from config."""

from __future__ import annotations

from cuda_opt_agent.core.config import LLMConfig
from cuda_opt_agent.core.exceptions import ConfigurationError
from cuda_opt_agent.llm.provider import LLMProvider


def create_provider(config: LLMConfig) -> LLMProvider:
    """Instantiate the configured LLM provider.

    Args:
        config: Full LLM configuration (provider selector + per-provider settings).

    Returns:
        Concrete LLMProvider instance.

    Raises:
        ConfigurationError: If provider name is unknown.
    """
    match config.provider:
        case "claude":
            from cuda_opt_agent.llm.claude_provider import ClaudeProvider
            return ClaudeProvider(config.claude)
        case "openai":
            from cuda_opt_agent.llm.openai_provider import OpenAIProvider
            return OpenAIProvider(config.openai)
        case "gemini":
            from cuda_opt_agent.llm.gemini_provider import GeminiProvider
            return GeminiProvider(config.gemini)
        case "ollama":
            from cuda_opt_agent.llm.ollama_provider import OllamaProvider
            return OllamaProvider(config.ollama)
        case _:
            raise ConfigurationError(
                f"Unknown LLM provider '{config.provider}'. "
                f"Supported: claude, openai, gemini, ollama"
            )
