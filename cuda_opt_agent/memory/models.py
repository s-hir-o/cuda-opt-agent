"""SQLAlchemy ORM models — one table per persisted domain object.

These are internal to the memory layer. External code uses Pydantic schemas only.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    JSON,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from cuda_opt_agent.memory.database import Base


# ── Kernel Sources ─────────────────────────────────────────────

class KernelSourceRow(Base):
    __tablename__ = "kernel_sources"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    language: Mapped[str] = mapped_column(String(50), nullable=False, default="cuda")
    source_code: Mapped[str] = mapped_column(Text, nullable=False)
    file_path: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    entry_point: Mapped[str] = mapped_column(String(255), nullable=False, default="main")
    submitted_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    metadata_json: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

    # Relationships
    variants: Mapped[list["KernelVariantRow"]] = relationship(
        back_populates="parent_kernel", cascade="all, delete-orphan"
    )
    runs: Mapped[list["OptimizationRunRow"]] = relationship(
        back_populates="kernel", cascade="all, delete-orphan"
    )


# ── Kernel Variants ────────────────────────────────────────────

class KernelVariantRow(Base):
    __tablename__ = "kernel_variants"

    id: Mapped[str] = mapped_column(String(255), primary_key=True)  # variant_id
    parent_kernel_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("kernel_sources.id", ondelete="CASCADE"), nullable=False, index=True
    )
    source_code: Mapped[str] = mapped_column(Text, nullable=False)
    language: Mapped[str] = mapped_column(String(50), nullable=False, default="cuda")
    entry_point: Mapped[str] = mapped_column(String(255), nullable=False, default="main")
    transform_history_json: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    iteration: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    metadata_json: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

    # Relationships
    parent_kernel: Mapped["KernelSourceRow"] = relationship(back_populates="variants")


# ── Optimization Runs ──────────────────────────────────────────

class OptimizationRunRow(Base):
    __tablename__ = "optimization_runs"

    id: Mapped[str] = mapped_column(String(255), primary_key=True)  # run_id
    kernel_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("kernel_sources.id", ondelete="CASCADE"), nullable=False, index=True
    )
    target_arch: Mapped[str] = mapped_column(String(50), nullable=False)
    max_iterations: Mapped[int] = mapped_column(Integer, nullable=False, default=20)
    best_variant_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    best_speedup: Mapped[float] = mapped_column(Float, nullable=False, default=1.0)
    started_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    status: Mapped[str] = mapped_column(
        String(50), nullable=False, default="running", index=True
    )
    config_snapshot_json: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    metadata_json: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

    # Relationships
    kernel: Mapped["KernelSourceRow"] = relationship(back_populates="runs")
    iterations: Mapped[list["OptimizationIterationRow"]] = relationship(
        back_populates="run", cascade="all, delete-orphan", order_by="OptimizationIterationRow.iteration_num"
    )


# ── Optimization Iterations ────────────────────────────────────

class OptimizationIterationRow(Base):
    __tablename__ = "optimization_iterations"
    __table_args__ = (
        Index("ix_iterations_run_iter", "run_id", "iteration_num", unique=True),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(
        String(255), ForeignKey("optimization_runs.id", ondelete="CASCADE"),
        nullable=False, index=True,
    )
    iteration_num: Mapped[int] = mapped_column(Integer, nullable=False)
    variant_id: Mapped[Optional[str]] = mapped_column(
        String(255), ForeignKey("kernel_variants.id", ondelete="SET NULL"), nullable=True,
    )
    started_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Inline JSON — lightweight pipeline data
    bottleneck_json: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    transform_spec_json: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    speedup_json: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    analysis_json: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    features_json: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    cost_estimate_json: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

    # FK references — detailed records stored in their own tables
    profile_metrics_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("profile_metrics.id", ondelete="SET NULL"), nullable=True,
    )
    compilation_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("compilation_results.id", ondelete="SET NULL"), nullable=True,
    )
    correctness_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("correctness_results.id", ondelete="SET NULL"), nullable=True,
    )
    benchmark_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("benchmark_results.id", ondelete="SET NULL"), nullable=True,
    )
    transform_result_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("transform_results.id", ondelete="SET NULL"), nullable=True,
    )

    # Relationships
    run: Mapped["OptimizationRunRow"] = relationship(back_populates="iterations")


# ── Profile Metrics ────────────────────────────────────────────

class ProfileMetricsRow(Base):
    __tablename__ = "profile_metrics"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    variant_id: Mapped[Optional[str]] = mapped_column(
        String(255), nullable=True, index=True,
    )
    kernel_name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    gpu_name: Mapped[str] = mapped_column(String(255), nullable=False, default="")
    arch: Mapped[str] = mapped_column(String(50), nullable=False, default="")

    # Timing
    gpu_time_ns: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    gpu_time_us: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Throughput
    sm_throughput: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    dram_throughput: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    l1_throughput: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    l2_throughput: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Occupancy
    achieved_occupancy: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    theoretical_occupancy: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Launch config
    registers_per_thread: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    block_size_json: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    grid_size_json: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    shared_mem_bytes: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    # Memory efficiency
    global_load_efficiency: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    global_store_efficiency: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    l2_hit_rate: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    l1_hit_rate: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Occupancy limiters
    occupancy_limited_by_registers: Mapped[bool] = mapped_column(Boolean, default=False)
    occupancy_limited_by_shared_mem: Mapped[bool] = mapped_column(Boolean, default=False)
    occupancy_limited_by_block_size: Mapped[bool] = mapped_column(Boolean, default=False)

    raw_metrics_json: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    profiled_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)


# ── Compilation Results ────────────────────────────────────────

class CompilationResultRow(Base):
    __tablename__ = "compilation_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    variant_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True, index=True)
    success: Mapped[bool] = mapped_column(Boolean, nullable=False)
    source_code: Mapped[str] = mapped_column(Text, nullable=False)
    arch: Mapped[str] = mapped_column(String(50), nullable=False)
    cubin_path: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    ptx_path: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    ptx_source: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    diagnostics_json: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    compile_time_s: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    compiled_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    stdout: Mapped[str] = mapped_column(Text, nullable=False, default="")
    stderr: Mapped[str] = mapped_column(Text, nullable=False, default="")
    returncode: Mapped[int] = mapped_column(Integer, nullable=False, default=0)


# ── Correctness Results ────────────────────────────────────────

class CorrectnessResultRow(Base):
    __tablename__ = "correctness_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    variant_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    baseline_variant_id: Mapped[str] = mapped_column(String(255), nullable=False)
    passed: Mapped[bool] = mapped_column(Boolean, nullable=False)
    max_absolute_error: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    max_relative_error: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    tolerance_absolute: Mapped[float] = mapped_column(Float, nullable=False, default=1e-5)
    tolerance_relative: Mapped[float] = mapped_column(Float, nullable=False, default=1e-4)
    failed_elements: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    total_elements: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    message: Mapped[str] = mapped_column(Text, nullable=False, default="")
    checked_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    metadata_json: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)


# ── Benchmark Results ──────────────────────────────────────────

class BenchmarkResultRow(Base):
    __tablename__ = "benchmark_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    variant_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    timing_json: Mapped[dict] = mapped_column(JSON, nullable=False)
    arch: Mapped[str] = mapped_column(String(50), nullable=False, default="")
    block_size_json: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    grid_size_json: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    correct: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    benchmarked_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    metadata_json: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)


# ── Transform Results ──────────────────────────────────────────

class TransformResultRow(Base):
    __tablename__ = "transform_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    spec_json: Mapped[dict] = mapped_column(JSON, nullable=False)
    original_source: Mapped[str] = mapped_column(Text, nullable=False)
    transformed_source: Mapped[str] = mapped_column(Text, nullable=False)
    success: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    warnings_json: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    diff_summary: Mapped[str] = mapped_column(Text, nullable=False, default="")
    applied_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    metadata_json: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)