"""Claude provider via the Anthropic SDK."""

from __future__ import annotations

import os
import time

from cuda_opt_agent.core.config import LLMProviderConfig
from cuda_opt_agent.core.exceptions import LLMError, LLMRateLimitError
from cuda_opt_agent.llm.provider import LLMProvider
from cuda_opt_agent.schemas.llm import LLMRequest, LLMResponse

try:
    import anthropic
except ImportError:
    anthropic = None  # type: ignore[assignment]


# Approximate pricing per million tokens (June 2025)
_INPUT_PRICE = 3.00
_OUTPUT_PRICE = 15.00


class ClaudeProvider(LLMProvider):

    def __init__(self, config: LLMProviderConfig) -> None:
        if anthropic is None:
            raise LLMError("anthropic package not installed (pip install anthropic)")
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            raise LLMError("ANTHROPIC_API_KEY environment variable not set")
        self._model = config.model or "claude-sonnet-4-20250514"
        self._max_tokens = config.max_tokens
        self._temperature = config.temperature
        self._client = anthropic.AsyncAnthropic(api_key=api_key)

    def provider_name(self) -> str:
        return "claude"

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        return (input_tokens * _INPUT_PRICE + output_tokens * _OUTPUT_PRICE) / 1_000_000

    async def health_check(self) -> bool:
        try:
            resp = await self._client.messages.create(
                model=self._model,
                max_tokens=10,
                messages=[{"role": "user", "content": "ping"}],
            )
            return len(resp.content) > 0
        except Exception:
            return False

    async def generate(self, request: LLMRequest) -> LLMResponse:
        t0 = time.monotonic()

        model = request.model or self._model
        max_tokens = request.max_tokens or self._max_tokens
        temperature = request.temperature if request.temperature is not None else self._temperature

        system = request.system_prompt or anthropic.NOT_GIVEN  # type: ignore[attr-defined]

        try:
            resp = await self._client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system,
                messages=[{"role": "user", "content": request.prompt}],
            )
        except anthropic.RateLimitError as exc:
            raise LLMRateLimitError(f"Claude rate limit: {exc}") from exc
        except anthropic.APIError as exc:
            raise LLMError(f"Claude API error: {exc}") from exc

        latency = time.monotonic() - t0
        text = resp.content[0].text if resp.content else ""
        usage = resp.usage

        return LLMResponse(
            content=text,
            model=model,
            provider="claude",
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cost_usd=self.estimate_cost(usage.input_tokens, usage.output_tokens),
            latency_s=latency,
        )
