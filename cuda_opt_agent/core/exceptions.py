"""All custom exception types for the CUDA optimization agent."""


class CudaOptError(Exception):
    """Base exception for all cuda-opt-agent errors."""


# ── Configuration ──────────────────────────────────────────────

class ConfigurationError(CudaOptError):
    """Invalid or missing configuration."""


# ── Compilation ────────────────────────────────────────────────

class CompilationError(CudaOptError):
    """NVCC compilation failed."""

    def __init__(self, message: str, stderr: str = "", returncode: int = 1) -> None:
        super().__init__(message)
        self.stderr = stderr
        self.returncode = returncode


class PTXExtractionError(CompilationError):
    """PTX extraction via nvcc -ptx failed."""


# ── Profiling ──────────────────────────────────────────────────

class ProfilingError(CudaOptError):
    """NCU profiling failed or returned unparseable output."""


class ProfilerUnavailableError(ProfilingError):
    """NCU binary not found or not functional."""


class MetricsParseError(ProfilingError):
    """Failed to parse NCU output into ProfileMetrics."""


# ── Analysis ───────────────────────────────────────────────────

class AnalysisError(CudaOptError):
    """Base for all analysis failures."""


class PTXParseError(AnalysisError):
    """Failed to parse PTX source."""


class DependencyGraphError(AnalysisError):
    """Failed to build dependency graph."""


# ── Transforms ─────────────────────────────────────────────────

class TransformError(CudaOptError):
    """Base for all transform failures."""


class TransformNotApplicableError(TransformError):
    """Transform is not applicable to the given kernel/bottleneck."""


class TransformValidationError(TransformError):
    """Transform output failed validation."""


# ── Benchmarking ───────────────────────────────────────────────

class BenchmarkError(CudaOptError):
    """Benchmark execution failed."""


class BenchmarkTimeoutError(BenchmarkError):
    """Benchmark exceeded timeout."""


# ── Correctness ────────────────────────────────────────────────

class CorrectnessError(CudaOptError):
    """Correctness check infrastructure error (not a test failure)."""


# ── LLM ────────────────────────────────────────────────────────

class LLMError(CudaOptError):
    """LLM provider error."""


class LLMRateLimitError(LLMError):
    """Rate limited by LLM provider."""


class LLMResponseParseError(LLMError):
    """LLM response did not match expected schema."""


# ── Database / Storage ─────────────────────────────────────────

class StorageError(CudaOptError):
    """Database or artifact store error."""


class ArtifactNotFoundError(StorageError):
    """Requested artifact does not exist."""
