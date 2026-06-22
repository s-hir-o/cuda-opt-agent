"""Kernel source and variant models."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class KernelLanguage(str, Enum):
    CUDA = "cuda"
    TRITON = "triton"
    PTX = "ptx"


class KernelSource(BaseModel):
    """An original kernel source file submitted for optimization."""

    name: str
    language: KernelLanguage = KernelLanguage.CUDA
    source_code: str
    file_path: str | None = None
    entry_point: str = "main"  # kernel function name
    submitted_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def short_name(self) -> str:
        """Name without extension."""
        return self.name.rsplit(".", 1)[0]


class KernelVariant(BaseModel):
    """A transformed version of a kernel, linked to its parent."""

    variant_id: str
    parent_source: KernelSource
    source_code: str
    language: KernelLanguage = KernelLanguage.CUDA
    entry_point: str = "main"
    transform_history: list[TransformSpec] = Field(default_factory=list)
    iteration: int = 0
    created_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)


# Avoid circular import — forward reference
from cuda_opt_agent.schemas.transform import TransformSpec  # noqa: E402

KernelVariant.model_rebuild()
