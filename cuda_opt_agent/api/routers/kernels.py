"""Kernel source CRUD endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from datetime import datetime

from cuda_opt_agent.api.dependencies import KernelRepoDep
from cuda_opt_agent.schemas.kernel import KernelLanguage, KernelSource

router = APIRouter()


# ── Request/Response models ────────────────────────────────────

class KernelSubmitRequest(BaseModel):
    name: str
    source_code: str
    language: str = "cuda"
    entry_point: str = "main"
    file_path: str | None = None


class KernelResponse(BaseModel):
    id: int
    name: str
    language: str
    entry_point: str
    source_code: str
    submitted_at: datetime

    model_config = {"from_attributes": True}


class KernelListResponse(BaseModel):
    kernels: list[KernelResponse]
    total: int


# ── Endpoints ──────────────────────────────────────────────────

@router.post("", response_model=KernelResponse, status_code=201)
async def submit_kernel(
    request: KernelSubmitRequest,
    repo: KernelRepoDep,
):
    """Submit a new kernel source for optimization."""
    try:
        lang = KernelLanguage(request.language)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported language '{request.language}'. Supported: cuda, triton, ptx",
        )

    kernel = KernelSource(
        name=request.name,
        language=lang,
        source_code=request.source_code,
        entry_point=request.entry_point,
        file_path=request.file_path,
    )

    db_id = repo.save_source(kernel)

    return KernelResponse(
        id=db_id,
        name=kernel.name,
        language=kernel.language.value,
        entry_point=kernel.entry_point,
        source_code=kernel.source_code,
        submitted_at=kernel.submitted_at,
    )


@router.get("", response_model=KernelListResponse)
async def list_kernels(
    repo: KernelRepoDep,
    offset: int = 0,
    limit: int = 50,
    name: str | None = None,
):
    """List submitted kernel sources. Optional filter by name."""
    if name:
        kernels = repo.find_sources_by_name(name)
    else:
        kernels = repo.list_sources(offset=offset, limit=limit)

    return KernelListResponse(
        kernels=[
            KernelResponse(
                id=i + 1,
                name=k.name,
                language=k.language.value,
                entry_point=k.entry_point,
                source_code=k.source_code,
                submitted_at=k.submitted_at,
            )
            for i, k in enumerate(kernels)
        ],
        total=len(kernels),
    )


@router.get("/{kernel_id}")
async def get_kernel(kernel_id: int, repo: KernelRepoDep):
    """Get a kernel source by ID."""
    kernel = repo.get_source(kernel_id)
    if kernel is None:
        raise HTTPException(status_code=404, detail=f"Kernel {kernel_id} not found")

    return KernelResponse(
        id=kernel_id,
        name=kernel.name,
        language=kernel.language.value,
        entry_point=kernel.entry_point,
        source_code=kernel.source_code,
        submitted_at=kernel.submitted_at,
    )
