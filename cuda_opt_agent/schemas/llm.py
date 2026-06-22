"""LLM request and response models."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class LLMRequest(BaseModel):
    """A request to an LLM provider."""

    prompt: str
    system_prompt: str = ""
    max_tokens: int = 4096
    temperature: float = 0.3
    model: str | None = None
    json_mode: bool = False
    retries: int = 3
    metadata: dict[str, Any] = Field(default_factory=dict)


class LLMResponse(BaseModel):
    """Response from an LLM provider."""

    content: str
    model: str = ""
    provider: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    latency_s: float = 0.0
    json_valid: bool | None = None  # None = not checked
    parsed_json: dict[str, Any] | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens
