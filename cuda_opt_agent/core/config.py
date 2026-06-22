"""Configuration loading and validation via Pydantic Settings + YAML."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


# ── Sub-section models ─────────────────────────────────────────

class ProjectConfig(BaseModel):
    name: str = "cuda-opt-agent"
    workspace_dir: str = "./workspace"
    artifact_dir: str = "./artifacts"


class CompilerConfig(BaseModel):
    nvcc_path: str = "nvcc"
    default_arch: str = "sm_80"
    extra_flags: list[str] = Field(default_factory=list)
    ptx_enabled: bool = True


class ProfilingMetricsConfig(BaseModel):
    pass  # metrics list is plain list[str] below


class ProfilingConfig(BaseModel):
    ncu_path: str = "ncu"
    enabled: bool = True
    metrics: list[str] = Field(default_factory=lambda: [
        "gpu__time_duration.sum",
        "sm__throughput.avg_pct_of_peak_sustained_elapsed",
        "dram__throughput.avg_pct_of_peak_sustained_elapsed",
        "sm__warps_active.avg.pct_of_peak_sustained_active",
        "launch__registers_per_thread",
    ])


class BenchmarkConfig(BaseModel):
    warmup_iterations: int = 3
    timing_iterations: int = 10
    use_cuda_events: bool = True
    timeout_seconds: int = 120


class LLMProviderConfig(BaseModel):
    model: str = ""
    max_tokens: int = 4096
    temperature: float = 0.3
    base_url: str | None = None


class LLMConfig(BaseModel):
    provider: Literal["claude", "openai", "gemini", "ollama"] = "claude"
    claude: LLMProviderConfig = Field(
        default_factory=lambda: LLMProviderConfig(model="claude-sonnet-4-20250514")
    )
    openai: LLMProviderConfig = Field(
        default_factory=lambda: LLMProviderConfig(model="gpt-4o")
    )
    gemini: LLMProviderConfig = Field(
        default_factory=lambda: LLMProviderConfig(model="gemini-2.5-pro")
    )
    ollama: LLMProviderConfig = Field(
        default_factory=lambda: LLMProviderConfig(model="codellama:34b")
    )


class CorrectnessToleranceConfig(BaseModel):
    relative: float = 1e-4
    absolute: float = 1e-5


class OptimizationConfig(BaseModel):
    max_iterations: int = 20
    convergence_patience: int = 3
    min_speedup_threshold: float = 1.01
    correctness_tolerance: CorrectnessToleranceConfig = Field(
        default_factory=CorrectnessToleranceConfig
    )


class GridSearchConfig(BaseModel):
    parallelism: int = 4


class RandomSearchConfig(BaseModel):
    max_samples: int = 50
    seed: int = 42


class SearchConfig(BaseModel):
    strategy: Literal["grid", "random", "bayesian"] = "grid"
    grid: GridSearchConfig = Field(default_factory=GridSearchConfig)
    random: RandomSearchConfig = Field(default_factory=RandomSearchConfig)


class AnalysisConfig(BaseModel):
    ptx_enabled: bool = True
    feature_extraction: bool = True
    cost_model: Literal["roofline", "occupancy"] = "roofline"


class DatabaseConfig(BaseModel):
    url: str = "sqlite:///./workspace/cuda_opt.db"
    echo: bool = False


class LoggingConfig(BaseModel):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    format: Literal["json", "console"] = "json"
    file: str | None = None


# ── Root settings ──────────────────────────────────────────────

class AppSettings(BaseSettings):
    """Root configuration. Loaded from YAML, overridable by env vars with CUDA_OPT_ prefix."""

    project: ProjectConfig = Field(default_factory=ProjectConfig)
    compiler: CompilerConfig = Field(default_factory=CompilerConfig)
    profiling: ProfilingConfig = Field(default_factory=ProfilingConfig)
    benchmark: BenchmarkConfig = Field(default_factory=BenchmarkConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    optimization: OptimizationConfig = Field(default_factory=OptimizationConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)
    analysis: AnalysisConfig = Field(default_factory=AnalysisConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    model_config = {
        "env_prefix": "CUDA_OPT_",
        "env_nested_delimiter": "__",
    }


def load_settings(config_path: str | Path | None = None) -> AppSettings:
    """Load settings from YAML file, falling back to defaults.

    Args:
        config_path: Path to YAML config. If None, uses default values.

    Returns:
        Validated AppSettings instance.
    """
    if config_path is None:
        return AppSettings()

    path = Path(config_path)
    if not path.exists():
        return AppSettings()

    with open(path) as f:
        raw: dict[str, Any] = yaml.safe_load(f) or {}

    return AppSettings(**raw)
