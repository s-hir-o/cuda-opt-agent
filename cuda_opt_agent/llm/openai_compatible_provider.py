"""OpenAI-compatible chat provider for locally served models.

This is intended for servers such as vLLM, TGI, or llama.cpp that expose
``/v1/chat/completions``. It keeps heavyweight CUDA inference dependencies out
of the optimizer process.
"""

from __future__ import annotations

import os
import time
from typing import Any

import httpx

from cuda_opt_agent.core.config import LLMProviderConfig
from cuda_opt_agent.core.exceptions import LLMError
from cuda_opt_agent.llm.provider import LLMProvider
from cuda_opt_agent.schemas.llm import LLMRequest, LLMResponse

_DEFAULT_BASE = "http://localhost:8000/v1"


class OpenAICompatibleProvider(LLMProvider):
    """Provider for OpenAI-compatible local inference endpoints."""

    def __init__(
        self,
        config: LLMProviderConfig,
        provider_name: str = "openai_compatible",
    ) -> None:
        self._provider_name = provider_name
        self._model = config.model or "ByteDance-Seed/cudaLLM-8B"
        self._max_tokens = config.max_tokens
        self._temperature = config.temperature
        self._base_url = (config.base_url or _DEFAULT_BASE).rstrip("/")
        self._api_key_env = config.api_key_env
        timeout = httpx.Timeout(
            connect=10.0,
            read=config.timeout_seconds,
            write=30.0,
            pool=10.0,
        )
        self._client = httpx.AsyncClient(timeout=timeout)

    def provider_name(self) -> str:
        return self._provider_name

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        return 0.0

    async def health_check(self) -> bool:
        try:
            resp = await self._client.get(
                f"{self._base_url}/models",
                headers=self._headers(),
            )
            return resp.status_code < 500
        except Exception:
            return False

    async def generate(self, request: LLMRequest) -> LLMResponse:
        t0 = time.monotonic()

        model = request.model or self._model
        max_tokens = request.max_tokens or self._max_tokens
        temperature = (
            request.temperature
            if request.temperature is not None
            else self._temperature
        )

        messages: list[dict[str, str]] = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        messages.append({"role": "user", "content": request.prompt})

        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }
        if request.json_mode:
            body["response_format"] = {"type": "json_object"}

        try:
            resp = await self._client.post(
                f"{self._base_url}/chat/completions",
                json=body,
                headers=self._headers(),
            )
        except httpx.ConnectError as exc:
            raise LLMError(
                f"Cannot reach {self._provider_name} endpoint at {self._base_url}"
            ) from exc
        except httpx.TimeoutException as exc:
            raise LLMError(
                f"{self._provider_name} request timed out after "
                f"{self._client.timeout.read} seconds"
            ) from exc

        if resp.status_code >= 400:
            raise LLMError(
                f"{self._provider_name} error {resp.status_code}: {resp.text}"
            )

        data = resp.json()
        latency = time.monotonic() - t0

        choices = data.get("choices", [])
        text = ""
        if choices:
            message = choices[0].get("message", {})
            text = message.get("content") or choices[0].get("text", "")

        usage = data.get("usage") or {}
        input_tok = int(usage.get("prompt_tokens", 0) or 0)
        output_tok = int(usage.get("completion_tokens", 0) or 0)

        return LLMResponse(
            content=text,
            model=model,
            provider=self._provider_name,
            input_tokens=input_tok,
            output_tokens=output_tok,
            cost_usd=0.0,
            latency_s=latency,
        )

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._api_key_env:
            api_key = os.environ.get(self._api_key_env, "")
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
        return headers
