"""Abstract Transformation base class and TransformBackend enum.

Every transform — text, AST, PTX, or LLVM — implements the same ABC.
The coordinator calls is_applicable() → apply() → validate(). Nothing else.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from cuda_opt_agent.schemas.bottleneck import BottleneckClass, BottleneckResult
from cuda_opt_agent.schemas.kernel import KernelSource, KernelVariant
from cuda_opt_agent.schemas.transform import TransformBackend, TransformResult

# NOTE: validate_transform is imported lazily in _base_validate() to avoid
# circular imports (base → validation → text/__init__ → text transforms → base).


class Transformation(ABC):
    """Base class for all kernel transforms.

    Subclasses implement six members:
    - is_applicable(): can this transform run on this kernel?
    - apply(): produce a TransformResult
    - validate(): sanity-check the output
    - name (property): unique identifier
    - targets (property): which bottleneck classes this addresses
    - parameter_schema (property): JSON Schema for params
    """

    @abstractmethod
    def is_applicable(
        self,
        kernel: KernelSource | KernelVariant,
        bottleneck: BottleneckResult,
        params: dict[str, Any],
    ) -> bool:
        """Return True if this transform can meaningfully run on the kernel."""

    @abstractmethod
    def apply(
        self,
        kernel: KernelSource | KernelVariant,
        params: dict[str, Any],
    ) -> TransformResult:
        """Apply the transform. Returns result with original + transformed source."""

    @abstractmethod
    def validate(self, result: TransformResult) -> bool:
        """Return True if the transformed source passes basic sanity checks."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique transform identifier (e.g., 'shared_memory_tiling')."""

    @property
    @abstractmethod
    def targets(self) -> list[BottleneckClass]:
        """Bottleneck classes this transform addresses."""

    @property
    @abstractmethod
    def parameter_schema(self) -> dict:
        """JSON Schema describing accepted parameters."""

    @property
    def backend(self) -> TransformBackend:
        """Default TEXT. Override in AST/PTX/LLVM subclasses."""
        return TransformBackend.TEXT
    def _make_result(
        self,
        kernel: KernelSource | KernelVariant,
        transformed: str,
        params: dict[str, Any],
        reason: str = "",
    ) -> TransformResult:
        """Helper to construct a TransformResult. Defaults to success=False.

        Callers (transform apply methods) must set result.success = True
        after a successful rewrite.
        """
        from cuda_opt_agent.schemas.transform import TransformSpec

        spec = TransformSpec(
            transform_name=self.name,
            backend=self.backend,
            params=params,
            reason=reason,
        )
        return TransformResult(
            spec=spec,
            original_source=kernel.source_code,
            transformed_source=transformed,
            success=False,
        )

    def _base_validate(self, result: TransformResult) -> bool:
        """Run the shared validation helper from validation.py."""
        from cuda_opt_agent.transforms.validation import validate_transform

        return validate_transform(
            result.transformed_source,
            original_source=result.original_source,
            result=result,
        )