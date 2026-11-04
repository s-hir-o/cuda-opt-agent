"""Prompt templates for LLM-driven CUDA optimization."""

from cuda_opt_agent.llm.prompts.generation import (
    GENERATION_RESPONSE_SCHEMA,
    build_generation_prompt,
)
from cuda_opt_agent.llm.prompts.transform import build_transform_prompt
from cuda_opt_agent.llm.prompts.schemas import TRANSFORM_RESPONSE_SCHEMA

__all__ = [
    "GENERATION_RESPONSE_SCHEMA",
    "TRANSFORM_RESPONSE_SCHEMA",
    "build_generation_prompt",
    "build_transform_prompt",
]
