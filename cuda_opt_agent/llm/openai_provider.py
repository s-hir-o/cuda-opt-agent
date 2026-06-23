"""OpenAI provider via the openai SDK."""

from __future__ import annotations

import os
import time

from cuda_opt_agent.core.config import LLMProviderConfig
from cuda_opt_agent.core.exceptions import LLMError, LLMRateLimitError
from cuda_opt_agent.llm.provider import LLMProvider
from cuda_opt_agent.schemas.llm import LLMRequest, LLMResponse

try:
    import openai
except ImportError:
    openai = None  # type: ignore[assignment]


# Approximate pricing per million tokens (GPT-4o, June 2025)
_INPUT_PRICE = 2.50
_OUTPUT_PRICE = 10.00


class OpenAIProvider(LLMProvider):

    def __init__(self, config: LLMProviderConfig) -> None:
        if openai is None:
            raise LLMError("openai package not installed (pip install openai)")
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            raise LLMError("OPENAI_API_KEY environment variable not set")
        self._model = config.model or "gpt-4o"
        self._max_tokens = config.max_tokens
        self._temperature = config.temperature
        self._client = openai.AsyncOpenAI(api_key=api_key)

    def provider_name(self) -> str:
        return "openai"

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        return (input_tokens * _INPUT_PRICE + output_tokens * _OUTPUT_PRICE) / 1_000_000

    async def health_check(self) -> bool:
        try:
            models = await self._client.models.list()
            return True
        except Exception:
            return False

    async def generate(self, request: LLMRequest) -> LLMResponse:
        t0 = time.monotonic()

        model = request.model or self._model
        max_tokens = request.max_tokens or self._max_tokens
        temperature = request.temperature if request.temperature is not None else self._temperature

        messages: list[dict[str, str]] = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        messages.append({"role": "user", "content": request.prompt})

        kwargs: dict = dict(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        if request.json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        try:
            resp = await self._client.chat.completions.create(**kwargs)
        except openai.RateLimitError as exc:
            raise LLMRateLimitError(f"OpenAI rate limit: {exc}") from exc
        except openai.APIError as exc:
            raise LLMError(f"OpenAI API error: {exc}") from exc

        latency = time.monotonic() - t0
        text = resp.choices[0].message.content or ""
        usage = resp.usage

        input_tok = usage.prompt_tokens if usage else 0
        output_tok = usage.completion_tokens if usage else 0

        return LLMResponse(
            content=text,
            model=model,
            provider="openai",
            input_tokens=input_tok,
            output_tokens=output_tok,
            cost_usd=self.estimate_cost(input_tok, output_tok),
            latency_s=latency,
        )
