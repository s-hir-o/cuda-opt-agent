
"""Tests for memory access pattern detection."""

from cuda_opt_agent.analysis.ptx.parser import parse_ptx
from cuda_opt_agent.analysis.memory.access_pattern import detect_access_patterns
from cuda_opt_agent.analysis.memory.shared_mem_analyzer import detect_bank_conflicts


STRIDED_PTX = """\
ld.global.f32 %f1, [%rd1+0];
ld.global.f32 %f2, [%rd1+4];
ld.global.f32 %f3, [%rd1+8];
ld.global.f32 %f4, [%rd1+12];
"""


class TestAccessPatternDetection:

    def test_stride_4(self):
        instrs = parse_ptx(STRIDED_PTX)
        patterns = detect_access_patterns(instrs)
        assert len(patterns) == 1
        assert patterns[0].pattern_type == "stride"
        assert patterns[0].stride_bytes == 4
        assert patterns[0].is_coalesced is True  # 4B = float

    def test_broadcast(self):
        ptx = "ld.global.f32 %f1, [%rd1+0];\nld.global.f32 %f2, [%rd1+0];\n"
        instrs = parse_ptx(ptx)
        patterns = detect_access_patterns(instrs)
        assert len(patterns) == 1
        assert patterns[0].pattern_type == "broadcast"

    def test_no_pattern_single_access(self):
        ptx = "ld.global.f32 %f1, [%rd1+0];\n"
        instrs = parse_ptx(ptx)
        patterns = detect_access_patterns(instrs)
        assert len(patterns) == 0  # Not enough accesses to detect pattern


class TestBankConflictDetection:

    def test_no_conflict_no_shared(self):
        ptx = "ld.global.f32 %f1, [%rd1];\n"
        instrs = parse_ptx(ptx)
        count, patterns = detect_bank_conflicts(instrs)
        assert count == 0

    def test_shared_access_no_conflict(self):
        ptx = "ld.shared.f32 %f1, [%rd1+0];\nld.shared.f32 %f2, [%rd1+4];\n"
        instrs = parse_ptx(ptx)
        count, patterns = detect_bank_conflicts(instrs)
        # Offsets 0 and 4 → banks 0 and 1 → no conflict
        assert count == 0
