"""Gemini provider via the REST API (httpx).

Uses Google's generateContent endpoint directly.
API key read from GEMINI_API_KEY or GOOGLE_API_KEY env var.
"""

from __future__ import annotations

import os
import time

import httpx

from cuda_opt_agent.core.config import LLMProviderConfig
from cuda_opt_agent.core.exceptions import LLMError, LLMRateLimitError
from cuda_opt_agent.llm.provider import LLMProvider
from cuda_opt_agent.schemas.llm import LLMRequest, LLMResponse

# Approximate pricing per million tokens (Gemini 2.5 Pro, June 2025)
_INPUT_PRICE = 1.25
_OUTPUT_PRICE = 10.00

_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
_TIMEOUT = httpx.Timeout(connect=10.0, read=120.0, write=10.0, pool=10.0)


class GeminiProvider(LLMProvider):

    def __init__(self, config: LLMProviderConfig) -> None:
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY", "")
        if not api_key:
            raise LLMError("GEMINI_API_KEY or GOOGLE_API_KEY environment variable not set")
        self._model = config.model or "gemini-2.5-pro"
        self._max_tokens = config.max_tokens
        self._temperature = config.temperature
        self._api_key = api_key
        self._client = httpx.AsyncClient(timeout=_TIMEOUT)

    def provider_name(self) -> str:
        return "gemini"

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        return (input_tokens * _INPUT_PRICE + output_tokens * _OUTPUT_PRICE) / 1_000_000

    async def health_check(self) -> bool:
        try:
            url = f"{_BASE_URL}/models/{self._model}"
            resp = await self._client.get(url, params={"key": self._api_key})
            return resp.status_code == 200
        except Exception:
            return False

    async def generate(self, request: LLMRequest) -> LLMResponse:
        t0 = time.monotonic()

        model = request.model or self._model
        max_tokens = request.max_tokens or self._max_tokens
        temperature = request.temperature if request.temperature is not None else self._temperature

        url = f"{_BASE_URL}/models/{model}:generateContent"

        body: dict = {
            "contents": [
                {"role": "user", "parts": [{"text": request.prompt}]}
            ],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            },
        }

        if request.system_prompt:
            body["systemInstruction"] = {"parts": [{"text": request.system_prompt}]}

        if request.json_mode:
            body["generationConfig"]["responseMimeType"] = "application/json"

        params = {"key": self._api_key}

        try:
            resp = await self._client.post(url, json=body, params=params)
        except httpx.ConnectError as exc:
            raise LLMError(f"Cannot reach Gemini API: {exc}") from exc

        if resp.status_code == 429:
            raise LLMRateLimitError(f"Gemini rate limit: {resp.text}")
        if resp.status_code >= 400:
            raise LLMError(f"Gemini API error {resp.status_code}: {resp.text}")

        data = resp.json()
        latency = time.monotonic() - t0

        # Extract text from response
        text = ""
        candidates = data.get("candidates", [])
        if candidates:
            parts = candidates[0].get("content", {}).get("parts", [])
            if parts:
                text = parts[0].get("text", "")

        # Extract token counts
        usage = data.get("usageMetadata", {})
        input_tok = usage.get("promptTokenCount", 0)
        output_tok = usage.get("candidatesTokenCount", 0)

        return LLMResponse(
            content=text,
            model=model,
            provider="gemini",
            input_tokens=input_tok,
            output_tokens=output_tok,
            cost_usd=self.estimate_cost(input_tok, output_tok),
            latency_s=latency,
        )
