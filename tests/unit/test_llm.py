"""Tests for LLM layer: provider ABC, JSON extraction, prompts, factory."""

from __future__ import annotations

import json

import pytest

from cuda_opt_agent.core.exceptions import ConfigurationError, LLMError, LLMRateLimitError
from cuda_opt_agent.llm.provider import LLMProvider, extract_json
from cuda_opt_agent.llm.factory import create_provider
from cuda_opt_agent.core.config import LLMConfig, LLMProviderConfig
from cuda_opt_agent.schemas.llm import LLMRequest, LLMResponse
from cuda_opt_agent.schemas.bottleneck import BottleneckClass, BottleneckResult
from cuda_opt_agent.schemas.kernel import KernelLanguage, KernelSource, KernelVariant
from cuda_opt_agent.schemas.metrics import ProfileMetrics
from cuda_opt_agent.llm.prompts.transform import build_transform_prompt
from cuda_opt_agent.llm.prompts.base import (
    format_bottleneck,
    format_kernel_source,
    format_metrics,
)


# ── JSON Extraction Tests ──────────────────────────────────────

class TestExtractJson:

    def test_direct_json_object(self):
        text = '{"transforms": [], "analysis_summary": "ok"}'
        result = extract_json(text)
        assert result is not None
        assert "transforms" in result

    def test_direct_json_array(self):
        text = '[1, 2, 3]'
        result = extract_json(text)
        assert result is not None
        assert result == [1, 2, 3]

    def test_json_in_markdown_fence(self):
        text = 'Here is the analysis:\n```json\n{"ok": true}\n```\nDone.'
        result = extract_json(text)
        assert result is not None
        assert result["ok"] is True

    def test_json_in_plain_fence(self):
        text = '```{"ok": true}\n```'
        result = extract_json(text)
        assert result is not None
        assert result["ok"] is True

    def test_json_embedded_in_text(self):
        text = 'Based on analysis, the result is {"speedup": 2.5} which is good.'
        result = extract_json(text)
        assert result is not None
        assert result["speedup"] == 2.5

    def test_json_array_embedded_in_text(self):
        """Verify the array regex correctly extracts embedded JSON arrays."""
        text = 'The results are [10, 20, 30] as expected.'
        result = extract_json(text)
        assert result is not None
        assert result == [10, 20, 30]

    def test_json_array_in_fence(self):
        text = 'Here are the items:\n```json\n["a", "b", "c"]\n```'
        result = extract_json(text)
        assert result is not None
        assert result == ["a", "b", "c"]

    def test_empty_string(self):
        assert extract_json("") is None
        assert extract_json("  ") is None

    def test_no_json(self):
        assert extract_json("This is just plain text with no JSON.") is None

    def test_invalid_json(self):
        assert extract_json("{not valid json}") is None

    def test_nested_json(self):
        obj = {
            "transforms": [
                {"name": "tiling", "params": {"tile_size": 16}},
                {"name": "unrolling", "params": {"factor": 4}},
            ],
            "warnings": ["may increase register pressure"],
        }
        text = f"Analysis complete:\n```json\n{json.dumps(obj)}\n```"
        result = extract_json(text)
        assert result is not None
        assert len(result["transforms"]) == 2
        assert result["transforms"][0]["params"]["tile_size"] == 16

    def test_prefers_direct_over_fence(self):
        """If the entire text is valid JSON, direct parse wins over fence extraction."""
        text = '{"direct": true}'
        result = extract_json(text)
        assert result is not None
        assert result["direct"] is True

    def test_handles_whitespace_around_fence(self):
        text = '  \n  ```json\n{"padded": true}\n```  \n  '
        result = extract_json(text)
        assert result is not None
        assert result["padded"] is True


# ── Mock Provider for generate_json Tests ─────────────────────

class _MockProvider(LLMProvider):
    """In-memory mock that returns pre-configured responses."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self._call_count = 0

    def provider_name(self) -> str:
        return "mock"

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        return 0.0

    async def health_check(self) -> bool:
        return True

    async def generate(self, request: LLMRequest) -> LLMResponse:
        content = self._responses[min(self._call_count, len(self._responses) - 1)]
        self._call_count += 1
        return LLMResponse(
            content=content,
            model="mock",
            provider="mock",
            input_tokens=10,
            output_tokens=20,
        )


class _FailingMockProvider(LLMProvider):
    """Mock that raises LLMError on generate()."""

    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    def provider_name(self) -> str:
        return "failing_mock"

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        return 0.0

    async def health_check(self) -> bool:
        return False

    async def generate(self, request: LLMRequest) -> LLMResponse:
        raise self._exc


class TestGenerateJson:

    @pytest.mark.asyncio
    async def test_valid_json_first_try(self):
        valid = json.dumps({"ok": True})
        provider = _MockProvider([valid])
        resp = await provider.generate_json(LLMRequest(prompt="test"))
        assert resp.json_valid is True
        assert resp.parsed_json == {"ok": True}
        assert provider._call_count == 1

    @pytest.mark.asyncio
    async def test_retry_on_invalid_json(self):
        provider = _MockProvider(["not json", '{"ok": true}'])
        resp = await provider.generate_json(LLMRequest(prompt="test"), retries=3)
        assert resp.json_valid is True
        assert resp.parsed_json == {"ok": True}
        assert provider._call_count == 2

    @pytest.mark.asyncio
    async def test_returns_last_response_on_exhaustion(self):
        provider = _MockProvider(["bad", "also bad", "still bad"])
        resp = await provider.generate_json(LLMRequest(prompt="test"), retries=3)
        assert resp.json_valid is False
        assert resp.parsed_json is None

    @pytest.mark.asyncio
    async def test_json_in_markdown_retry_succeeds(self):
        provider = _MockProvider([
            "no json here",
            '```json\n{"recovered": true}\n```',
        ])
        resp = await provider.generate_json(LLMRequest(prompt="test"), retries=2)
        assert resp.json_valid is True
        assert resp.parsed_json["recovered"] is True

    @pytest.mark.asyncio
    async def test_retry_failure_with_provider_error(self):
        """Provider exception (LLMError) propagates without returning a response."""
        provider = _FailingMockProvider(LLMError("API connection failed"))
        with pytest.raises(LLMError, match="API connection failed"):
            await provider.generate_json(LLMRequest(prompt="test"), retries=3)

    @pytest.mark.asyncio
    async def test_retry_failure_with_rate_limit(self):
        """Rate-limit error propagates as LLMRateLimitError (subclass of LLMError)."""
        provider = _FailingMockProvider(LLMRateLimitError("Rate limit exceeded"))
        with pytest.raises(LLMRateLimitError, match="Rate limit exceeded"):
            await provider.generate_json(LLMRequest(prompt="test"), retries=3)

    @pytest.mark.asyncio
    async def test_system_prompt_not_mutated(self):
        """Original request's system_prompt is unchanged after generate_json."""
        req = LLMRequest(prompt="test", system_prompt="Original prompt")
        provider = _MockProvider(["not json", '{"ok": true}'])
        await provider.generate_json(req, retries=3)
        assert req.system_prompt == "Original prompt"

    @pytest.mark.asyncio
    async def test_array_response_wrapped_in_items(self):
        """generate_json wraps array responses in {"items": ...}."""
        provider = _MockProvider(['[1, 2, 3]'])
        resp = await provider.generate_json(LLMRequest(prompt="test"))
        assert resp.json_valid is True
        assert resp.parsed_json == {"items": [1, 2, 3]}

    @pytest.mark.asyncio
    async def test_retries_zero_still_runs_once(self):
        """retries=0 clamps to 1 attempt via max(1, retries)."""
        provider = _MockProvider(['{"ok": true}'])
        resp = await provider.generate_json(LLMRequest(prompt="test"), retries=0)
        assert resp.json_valid is True
        assert provider._call_count == 1


# ── Prompt Builder Tests ───────────────────────────────────────

class TestPromptBuilders:

    @pytest.fixture
    def kernel(self) -> KernelSource:
        return KernelSource(
            name="matmul.cu",
            language=KernelLanguage.CUDA,
            source_code="__global__ void matmul() {}",
            entry_point="matmul",
        )

    @pytest.fixture
    def bottleneck(self) -> BottleneckResult:
        return BottleneckResult(
            primary=BottleneckClass.MEMORY_BOUND,
            confidence=0.85,
            reasoning="DRAM throughput > SM throughput",
            recommended_transforms=["shared_memory_tiling"],
        )

    @pytest.fixture
    def metrics(self) -> ProfileMetrics:
        return ProfileMetrics(
            kernel_name="matmul",
            gpu_time_us=150.0,
            sm_throughput=45.0,
            dram_throughput=82.0,
            achieved_occupancy=0.625,
            registers_per_thread=40,
            block_size=(16, 16, 1),
            occupancy_limited_by_registers=True,
        )

    def test_format_kernel_source(self, kernel):
        text = format_kernel_source(kernel)
        assert "matmul.cu" in text
        assert "cuda" in text
        assert "__global__ void matmul" in text
        assert "```cuda" in text

    def test_format_bottleneck(self, bottleneck):
        text = format_bottleneck(bottleneck)
        assert "memory_bound" in text
        assert "85%" in text
        assert "DRAM" in text
        assert "shared_memory_tiling" in text

    def test_format_metrics(self, metrics):
        text = format_metrics(metrics)
        assert "150.00" in text
        assert "82.0%" in text
        assert "62.5%" in text
        assert "40" in text
        assert "registers" in text.lower()

    def test_build_transform_prompt_structure(self, kernel, bottleneck, metrics):
        request = build_transform_prompt(kernel, bottleneck, metrics)

        assert request.json_mode is True
        assert len(request.system_prompt) > 100
        assert "shared_memory_tiling" in request.system_prompt
        assert "loop_unrolling" in request.system_prompt
        assert "matmul.cu" in request.prompt
        assert "memory_bound" in request.prompt
        assert "82.0%" in request.prompt

    def test_build_transform_prompt_without_metrics(self, kernel, bottleneck):
        request = build_transform_prompt(kernel, bottleneck)
        assert "matmul.cu" in request.prompt
        assert request.json_mode is True

    def test_variant_prompt_includes_history(self, kernel, bottleneck):
        variant = KernelVariant(
            variant_id="v1_tiled",
            parent_source=kernel,
            source_code="// optimized",
            transform_history=[
                __import__(
                    "cuda_opt_agent.schemas.transform", fromlist=["TransformSpec"]
                ).TransformSpec(
                    transform_name="shared_memory_tiling",
                    params={"tile_size": 16},
                    reason="memory bound",
                ),
            ],
        )
        request = build_transform_prompt(variant, bottleneck)
        assert "v1_tiled" in request.prompt
        assert "shared_memory_tiling" in request.prompt
        assert "tile_size" in request.prompt


# ── Factory Tests ──────────────────────────────────────────────

class TestFactory:

    def test_unknown_provider_raises(self):
        """Invalid provider is rejected — Pydantic catches Literal violations
        at construction time, which is even better than the factory rejecting
        it later.  Either outcome satisfies the intent of this test."""
        from pydantic import ValidationError as PydanticValidationError

        with pytest.raises((ConfigurationError, PydanticValidationError)):
            config = LLMConfig(provider="nonexistent")  # type: ignore[arg-type]
            create_provider(config)

    def test_claude_requires_api_key(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        config = LLMConfig(provider="claude")
        with pytest.raises(LLMError, match="ANTHROPIC_API_KEY"):
            create_provider(config)

    def test_openai_requires_api_key(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        config = LLMConfig(provider="openai")
        with pytest.raises(LLMError, match="OPENAI_API_KEY"):
            create_provider(config)

    def test_gemini_requires_api_key(self, monkeypatch):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        config = LLMConfig(provider="gemini")
        with pytest.raises(LLMError, match="GEMINI_API_KEY"):
            create_provider(config)

    def test_ollama_creation(self):
        config = LLMConfig(provider="ollama")
        provider = create_provider(config)
        assert provider.provider_name() == "ollama"
        assert provider.estimate_cost(1000, 1000) == 0.0

