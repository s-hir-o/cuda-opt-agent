
"""Tests for PTX parser and instruction classification."""

import pytest

from cuda_opt_agent.analysis.ptx.parser import parse_ptx
from cuda_opt_agent.analysis.ptx.instruction_analyzer import classify_opcode
from cuda_opt_agent.schemas.analysis import OpcodeClass


SAMPLE_PTX = """\
.entry test_kernel(.param .u64 A, .param .u64 B, .param .u64 C)
{
    .reg .f32 %f<8>;
    .reg .b64 %rd<4>;

    ld.param.u64 %rd1, [A];
    ld.param.u64 %rd2, [B];
    ld.param.u64 %rd3, [C];
    ld.global.f32 %f1, [%rd1];
    ld.global.f32 %f2, [%rd2];
    add.f32 %f3, %f1, %f2;
    st.global.f32 [%rd3], %f3;
    ret;
}
"""


class TestParsePTX:

    def test_instruction_count(self):
        instrs = parse_ptx(SAMPLE_PTX)
        # Actually: ld.param×3, ld.global×2, add×1, st×1, ret×1 = 8
        assert len(instrs) == 8

    def test_indices_sequential(self):
        instrs = parse_ptx(SAMPLE_PTX)
        assert [i.index for i in instrs] == list(range(8))

    def test_add_classification(self):
        instrs = parse_ptx(SAMPLE_PTX)
        add_instr = instrs[5]  # add.f32
        assert add_instr.opcode == "add.f32"
        assert add_instr.opcode_class == OpcodeClass.ALU
        assert "%f3" in add_instr.dest_registers
        assert "%f1" in add_instr.src_registers
        assert "%f2" in add_instr.src_registers

    def test_load_global(self):
        instrs = parse_ptx(SAMPLE_PTX)
        ld = instrs[3]  # ld.global.f32 %f1, [%rd1]
        assert ld.opcode_class == OpcodeClass.MEMORY_LOAD
        assert ld.memory_space == "global"
        assert "%f1" in ld.dest_registers
        assert "%rd1" in ld.src_registers

    def test_store_global(self):
        instrs = parse_ptx(SAMPLE_PTX)
        st = instrs[6]  # st.global.f32 [%rd3], %f3
        assert st.opcode_class == OpcodeClass.MEMORY_STORE
        assert st.memory_space == "global"
        assert st.dest_registers == []
        assert "%f3" in st.src_registers
        assert "%rd3" in st.src_registers

    def test_ret_no_registers(self):
        instrs = parse_ptx(SAMPLE_PTX)
        ret = instrs[7]
        assert ret.opcode == "ret"
        assert ret.is_branch
        assert ret.dest_registers == []
        assert ret.src_registers == []

    def test_directives_skipped(self):
        ptx = ".version 7.8\n.target sm_80\n.reg .u32 %r<4>;\nadd.s32 %r1, %r2, %r3;\n"
        instrs = parse_ptx(ptx)
        assert len(instrs) == 1
        assert instrs[0].opcode == "add.s32"

    def test_labels_skipped(self):
        ptx = "BB0_1:\nadd.s32 %r1, %r2, %r3;\nBB0_2:\nret;\n"
        instrs = parse_ptx(ptx)
        assert len(instrs) == 2

    def test_predicate(self):
        ptx = "@%p1 bra BB0_1;\nadd.s32 %r1, %r2, %r3;\n"
        instrs = parse_ptx(ptx)
        assert instrs[0].predicate == "@%p1"
        assert instrs[0].is_branch

    def test_negated_predicate(self):
        ptx = "@!%p1 bra EXIT;\nret;\n"
        instrs = parse_ptx(ptx)
        assert instrs[0].predicate == "@!%p1"

    def test_memory_offset(self):
        ptx = "ld.global.f32 %f1, [%rd1+16];\n"
        instrs = parse_ptx(ptx)
        assert instrs[0].memory_offset == 16

    def test_empty_input(self):
        assert parse_ptx("") == []
        assert parse_ptx("// just a comment\n") == []

    def test_block_comments_stripped(self):
        ptx = "/* comment */ add.s32 %r1, %r2, %r3;\n"
        instrs = parse_ptx(ptx)
        assert len(instrs) == 1


class TestClassifyOpcode:

    def test_alu(self):
        assert classify_opcode("add.s32") == OpcodeClass.ALU
        assert classify_opcode("sub.f32") == OpcodeClass.ALU
        assert classify_opcode("shl.b32") == OpcodeClass.ALU
        assert classify_opcode("and.b32") == OpcodeClass.ALU

    def test_fma(self):
        assert classify_opcode("fma.rn.f32") == OpcodeClass.FMA
        assert classify_opcode("mad.lo.s32") == OpcodeClass.FMA

    def test_memory_load(self):
        assert classify_opcode("ld.global.f32") == OpcodeClass.MEMORY_LOAD
        assert classify_opcode("ldu.u32") == OpcodeClass.MEMORY_LOAD

    def test_memory_store(self):
        assert classify_opcode("st.global.f32") == OpcodeClass.MEMORY_STORE

    def test_shared_load(self):
        assert classify_opcode("ld.shared.f32") == OpcodeClass.SHARED_LOAD
        assert classify_opcode("lds.u32") == OpcodeClass.SHARED_LOAD

    def test_shared_store(self):
        assert classify_opcode("st.shared.f32") == OpcodeClass.SHARED_STORE

    def test_control_flow(self):
        assert classify_opcode("bra") == OpcodeClass.CONTROL_FLOW
        assert classify_opcode("ret") == OpcodeClass.CONTROL_FLOW
        assert classify_opcode("call") == OpcodeClass.CONTROL_FLOW

    def test_comparison(self):
        assert classify_opcode("setp.lt.f32") == OpcodeClass.COMPARISON
        assert classify_opcode("setp.eq.s32") == OpcodeClass.COMPARISON

    def test_conversion(self):
        assert classify_opcode("cvt.rn.f32.u32") == OpcodeClass.CONVERSION

    def test_atomic(self):
        assert classify_opcode("atom.global.add.f32") == OpcodeClass.ATOMIC

    def test_sync(self):
        assert classify_opcode("bar.sync") == OpcodeClass.SYNC

    def test_mov_as_alu(self):
        assert classify_opcode("mov.u32") == OpcodeClass.ALU
