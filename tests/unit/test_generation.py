"""Tests for LLM-driven CUDA generation."""

from __future__ import annotations

import json

import pytest

from cuda_opt_agent.agents.base import AgentContext
from cuda_opt_agent.agents.generator_agent import GeneratorAgent
from cuda_opt_agent.llm.prompts.generation import build_generation_prompt
from cuda_opt_agent.llm.provider import LLMProvider
from cuda_opt_agent.schemas.bottleneck import BottleneckClass, BottleneckResult
from cuda_opt_agent.schemas.kernel import KernelLanguage, KernelSource
from cuda_opt_agent.schemas.llm import LLMRequest, LLMResponse
from cuda_opt_agent.schemas.metrics import ProfileMetrics


class _MockProvider(LLMProvider):
    def __init__(self, content: str) -> None:
        self._content = content
        self.last_request: LLMRequest | None = None

    def provider_name(self) -> str:
        return "mock"

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        return 0.0

    async def health_check(self) -> bool:
        return True

    async def generate(self, request: LLMRequest) -> LLMResponse:
        self.last_request = request
        return LLMResponse(
            content=self._content,
            model="mock-model",
            provider="mock",
            input_tokens=10,
            output_tokens=20,
        )


@pytest.fixture
def kernel() -> KernelSource:
    return KernelSource(
        name="scale.cu",
        language=KernelLanguage.CUDA,
        source_code=(
            "__global__ void scale(float* x, float a) { "
            "int i = threadIdx.x; x[i] = a * x[i]; }"
        ),
        entry_point="scale",
    )


@pytest.fixture
def bottleneck() -> BottleneckResult:
    return BottleneckResult(
        primary=BottleneckClass.MEMORY_BOUND,
        confidence=0.8,
        reasoning="DRAM throughput dominates",
        recommended_transforms=["vectorized_loads"],
    )


def test_build_generation_prompt_requests_complete_cuda(kernel, bottleneck):
    metrics = ProfileMetrics(
        kernel_name="scale",
        gpu_time_us=12.0,
        dram_throughput=80.0,
        sm_throughput=35.0,
    )

    request = build_generation_prompt(
        kernel=kernel,
        bottleneck=bottleneck,
        metrics=metrics,
        max_candidates=1,
    )

    assert request.json_mode is True
    assert "complete CUDA source" in request.system_prompt
    assert "Generate up to 1 complete CUDA kernel" in request.prompt
    assert "__global__ void scale" in request.prompt
    assert "memory_bound" in request.prompt


@pytest.mark.asyncio
async def test_generator_agent_returns_valid_candidate(kernel, bottleneck):
    generated_source = (
        "__global__ void scale_opt(float* x, float a) { "
        "int i = blockIdx.x * blockDim.x + threadIdx.x; x[i] = a * x[i]; }"
    )
    content = json.dumps({
        "candidates": [{
            "source_code": generated_source,
            "entry_point": "scale_opt",
            "rationale": "Use global index for full-grid coverage.",
            "assumptions": ["x has enough elements for launched threads"],
            "launch_config": {"block_size": 256},
            "expected_improvement": 1.1,
            "risks": ["No bounds guard without problem size"],
        }]
    })
    provider = _MockProvider(content)
    agent = GeneratorAgent(provider)
    ctx = AgentContext(run_id="run_1", kernel=kernel, arch="sm_80", bottleneck=bottleneck)

    candidates = await agent.run(ctx, max_candidates=1)

    assert len(candidates) == 1
    assert candidates[0].entry_point == "scale_opt"
    assert candidates[0].provider == "mock"
    assert candidates[0].model == "mock-model"
    assert provider.last_request is not None
    assert provider.last_request.json_mode is True


@pytest.mark.asyncio
async def test_generator_agent_rejects_missing_entry_point(kernel, bottleneck):
    content = json.dumps({
        "candidates": [{
            "source_code": "__global__ void other(float* x) { x[threadIdx.x] = 0; }",
            "entry_point": "scale_opt",
        }]
    })
    agent = GeneratorAgent(_MockProvider(content))
    ctx = AgentContext(run_id="run_1", kernel=kernel, arch="sm_80", bottleneck=bottleneck)

    candidates = await agent.run(ctx, max_candidates=1)

    assert candidates == []
