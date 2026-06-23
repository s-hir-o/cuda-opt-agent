"""Abstract LLM provider with concrete JSON-generation retry logic.

Each provider implements generate(). The base class handles generate_json()
with extraction + retry so that logic is never duplicated.
"""

from __future__ import annotations

import json
import re
import time
from abc import ABC, abstractmethod
from typing import Any

from cuda_opt_agent.core.exceptions import LLMError, LLMResponseParseError
from cuda_opt_agent.schemas.llm import LLMRequest, LLMResponse


# ── JSON extraction helper ─────────────────────────────────────

_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*\n?(.*?)\n?\s*```", re.DOTALL)
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)
_JSON_ARRAY_RE = re.compile(r"\[.*\]", re.DOTALL)


def extract_json(text: str) -> dict[str, Any] | list[Any] | None:
    """Best-effort extraction of a JSON value from LLM output.

    Tries, in order:
    1. Direct parse of entire text.
    2. Parse content inside ```json ... ``` or ``` ... ``` fences.
    3. First balanced { ... } or [ ... ] substring.
    """
    text = text.strip()
    if not text:
        return None

    # 1. Direct
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 2. Fenced block
    m = _JSON_BLOCK_RE.search(text)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

    # 3. First JSON object / array
    for regex in (_JSON_OBJECT_RE, _JSON_ARRAY_RE):
        m = regex.search(text)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                continue

    return None


# ── Abstract Base ──────────────────────────────────────────────

class LLMProvider(ABC):
    """Base class for all LLM providers.

    Subclasses implement: generate(), health_check(), estimate_cost(), provider_name().
    The base class provides generate_json() with automatic retry on parse failure.
    """

    # ── Abstract interface ─────────────────────────────────────

    @abstractmethod
    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate a text completion. Raises LLMError on API failure."""

    @abstractmethod
    async def health_check(self) -> bool:
        """Return True if the provider is reachable and credentials are valid."""

    @abstractmethod
    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Estimate USD cost for the given token counts."""

    @abstractmethod
    def provider_name(self) -> str:
        """Provider identifier string (e.g., 'claude', 'openai')."""

    # ── Concrete: JSON generation with retry ───────────────────

    async def generate_json(
        self, request: LLMRequest, retries: int = 3
    ) -> LLMResponse:
        """Generate and parse JSON output. Retries on parse failure.

        On first attempt, sends the request as-is with json_mode=True.
        On subsequent attempts, strengthens the JSON instruction in the system prompt.
        Original system prompt is preserved; retry instructions are applied to copies only.
        """
        last_response: LLMResponse | None = None
        original_system = request.system_prompt

        for attempt in range(max(1, retries)):
            req = request.model_copy()

            if attempt > 0:
                suffix = (
                    "\n\nIMPORTANT: Your previous response contained invalid JSON. "
                    "Respond with ONLY a valid JSON object. No markdown fences, "
                    "no explanation, no text outside the JSON."
                )
                req.system_prompt = (original_system or "") + suffix

            req.json_mode = True
            response = await self.generate(req)

            parsed = extract_json(response.content)
            if parsed is not None and isinstance(parsed, (dict, list)):
                response.json_valid = True
                response.parsed_json = parsed if isinstance(parsed, dict) else {"items": parsed}
                return response

            response.json_valid = False
            last_response = response

        if last_response is not None:
            return last_response

        raise LLMError("generate_json produced no response")