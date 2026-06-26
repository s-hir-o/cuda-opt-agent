"""Tests for dependency graph construction."""

from cuda_opt_agent.analysis.ptx.parser import parse_ptx
from cuda_opt_agent.analysis.ptx.dependency_graph import build_dependency_graph
from cuda_opt_agent.schemas.analysis import DepType


SIMPLE_PTX = """\
ld.param.u64 %rd1, [A];
ld.param.u64 %rd2, [B];
ld.global.f32 %f1, [%rd1];
ld.global.f32 %f2, [%rd2];
add.f32 %f3, %f1, %f2;
st.global.f32 [%rd2], %f3;
ret;
"""


class TestDependencyGraph:

    def test_raw_dependencies(self):
        instrs = parse_ptx(SIMPLE_PTX)
        graph = build_dependency_graph(instrs)

        raw_edges = [e for e in graph.edges if e.dep_type == DepType.RAW]
        # Expected RAW: 0→2 (%rd1), 1→3 (%rd2), 2→4 (%f1), 3→4 (%f2), 4→5 (%f3)
        assert len(raw_edges) == 6

        # Verify specific edges
        raw_pairs = {(e.from_idx, e.to_idx) for e in raw_edges}
        assert (0, 2) in raw_pairs  # %rd1: ld.param → ld.global
        assert (1, 3) in raw_pairs  # %rd2: ld.param → ld.global
        assert (2, 4) in raw_pairs  # %f1: ld.global → add
        assert (3, 4) in raw_pairs  # %f2: ld.global → add
        assert (4, 5) in raw_pairs  # %f3: add → st

    def test_no_waw_in_simple_case(self):
        instrs = parse_ptx(SIMPLE_PTX)
        graph = build_dependency_graph(instrs)
        waw_edges = [e for e in graph.edges if e.dep_type == DepType.WAW]
        assert len(waw_edges) == 0

    def test_no_war_in_simple_case(self):
        instrs = parse_ptx(SIMPLE_PTX)
        graph = build_dependency_graph(instrs)
        war_edges = [e for e in graph.edges if e.dep_type == DepType.WAR]
        assert len(war_edges) == 0

    def test_critical_path_length(self):
        instrs = parse_ptx(SIMPLE_PTX)
        graph = build_dependency_graph(instrs)
        # Longest chain: ld.param → ld.global → add → st = 3 hops
        assert graph.critical_path_length == 3

    def test_ilp_estimate(self):
        instrs = parse_ptx(SIMPLE_PTX)
        graph = build_dependency_graph(instrs)
        # 7 instructions, critical path 3 → ILP ≈ 7/3
        assert abs(graph.estimated_ilp - 7 / 3) < 0.01

    def test_empty_input(self):
        graph = build_dependency_graph([])
        assert graph.num_instructions == 0
        assert graph.critical_path_length == 0
        assert graph.edge_count == 0

    def test_war_dependency(self):
        # WAR: instruction reads %r1, later instruction writes %r1
        ptx = "ld.global.u32 %r1, [%rd1];\nst.global.u32 [%rd2], %r1;\nmov.u32 %r1, 0;\n"
        instrs = parse_ptx(ptx)
        graph = build_dependency_graph(instrs)
        war = [e for e in graph.edges if e.dep_type == DepType.WAR]
        # mov.u32 writes %r1, which st reads
        assert any(e.register == "%r1" for e in war)

    def test_waw_dependency(self):
        # WAW: two instructions write same register
        ptx = "mov.u32 %r1, 0;\nmov.u32 %r1, 1;\n"
        instrs = parse_ptx(ptx)
        graph = build_dependency_graph(instrs)
        waw = [e for e in graph.edges if e.dep_type == DepType.WAW]
        assert len(waw) == 1
        assert waw[0].register == "%r1"


