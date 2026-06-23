"""Ollama provider via the local REST API (httpx).

Targets the /api/chat endpoint. No API key required.
"""

from __future__ import annotations

import time

import httpx

from cuda_opt_agent.core.config import LLMProviderConfig
from cuda_opt_agent.core.exceptions import LLMError
from cuda_opt_agent.llm.provider import LLMProvider
from cuda_opt_agent.schemas.llm import LLMRequest, LLMResponse

_DEFAULT_BASE = "http://localhost:11434"
_TIMEOUT = httpx.Timeout(connect=5.0, read=300.0, write=10.0, pool=5.0)


class OllamaProvider(LLMProvider):

    def __init__(self, config: LLMProviderConfig) -> None:
        self._model = config.model or "codellama:34b"
        self._max_tokens = config.max_tokens
        self._temperature = config.temperature
        self._base_url = (config.base_url or _DEFAULT_BASE).rstrip("/")
        self._client = httpx.AsyncClient(timeout=_TIMEOUT)

    def provider_name(self) -> str:
        return "ollama"

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        return 0.0  # Local inference is free

    async def health_check(self) -> bool:
        try:
            resp = await self._client.get(f"{self._base_url}/api/tags")
            return resp.status_code == 200
        except Exception:
            return False

    async def generate(self, request: LLMRequest) -> LLMResponse:
        t0 = time.monotonic()

        model = request.model or self._model
        temperature = request.temperature if request.temperature is not None else self._temperature

        messages: list[dict[str, str]] = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        messages.append({"role": "user", "content": request.prompt})

        body: dict = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": request.max_tokens or self._max_tokens,
            },
        }
        if request.json_mode:
            body["format"] = "json"

        url = f"{self._base_url}/api/chat"

        try:
            resp = await self._client.post(url, json=body)
        except httpx.ConnectError as exc:
            raise LLMError(
                f"Cannot reach Ollama at {self._base_url}. Is it running?"
            ) from exc

        if resp.status_code >= 400:
            raise LLMError(f"Ollama error {resp.status_code}: {resp.text}")

        data = resp.json()
        latency = time.monotonic() - t0

        text = data.get("message", {}).get("content", "")
        prompt_tok = data.get("prompt_eval_count", 0)
        output_tok = data.get("eval_count", 0)

        return LLMResponse(
            content=text,
            model=model,
            provider="ollama",
            input_tokens=prompt_tok,
            output_tokens=output_tok,
            cost_usd=0.0,
            latency_s=latency,
        )
