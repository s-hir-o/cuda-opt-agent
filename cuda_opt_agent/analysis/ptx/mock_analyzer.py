"""Mock kernel analyzer for testing without PTX toolchain."""

from __future__ import annotations

from cuda_opt_agent.analysis.base import KernelAnalyzer
from cuda_opt_agent.analysis.ptx.parser import parse_ptx
from cuda_opt_agent.analysis.ptx.dependency_graph import build_dependency_graph
from cuda_opt_agent.analysis.memory.access_pattern import detect_access_patterns
from cuda_opt_agent.analysis.memory.shared_mem_analyzer import detect_bank_conflicts
from cuda_opt_agent.schemas.analysis import KernelAnalysis
from cuda_opt_agent.schemas.kernel import KernelSource, KernelVariant


class MockPTXAnalyzer(KernelAnalyzer):
    """Parses real PTX if provided, otherwise returns empty analysis."""

    @property
    def name(self) -> str:
        return "ptx_mock"

    async def is_available(self) -> bool:
        return True

    async def analyze(
        self,
        kernel: KernelSource | KernelVariant,
        ptx_source: str | None,
        run_id: str,
    ) -> KernelAnalysis:
        if ptx_source is None:
            return KernelAnalysis(analysis_backend="mock_empty")

        instructions = parse_ptx(ptx_source)
        dep_graph = build_dependency_graph(instructions)
        access_patterns = detect_access_patterns(instructions)
        bank_conflicts, bank_patterns = detect_bank_conflicts(instructions)

        # Instruction mix
        mix: dict[str, int] = {}
        for instr in instructions:
            key = instr.opcode_class.value
            mix[key] = mix.get(key, 0) + 1

        return KernelAnalysis(
            instructions=instructions,
            dependency_graph=dep_graph,
            access_patterns=access_patterns + bank_patterns,
            total_instructions=len(instructions),
            instruction_mix=mix,
            critical_path_length=dep_graph.critical_path_length,
            ilp_estimate=dep_graph.estimated_ilp,
            has_shared_memory=any(i.memory_space == "shared" for i in instructions),
            has_divergent_branches=any(
                i.predicate and i.is_branch for i in instructions
            ),
            bank_conflict_count=bank_conflicts,
            analysis_backend="ptx_mock",
        )


