from __future__ import annotations

from dataclasses import asdict

import pytest

from tesseract.vm.ir import Instruction
from tesseract.vm.machine import Trap, VM
from tesseract.vm.state import VMState


@pytest.mark.parametrize(
    ("opcode", "expected"),
    [
        ("ADD", 10),
        ("SUB", 4),
        ("MUL", 21),
        ("DIV", 2),
    ],
)
def test_arithmetic_opcodes(vm: VM, opcode: str, expected: int) -> None:
    program = [
        Instruction("CONST", dst=0, imm=7),
        Instruction("CONST", dst=1, imm=3),
        Instruction(opcode, dst=2, src1=0, src2=1),
        Instruction("HALT"),
    ]

    state = vm.execute(program)

    assert state.registers[2] == expected
    assert state.halted is True
    assert state.halt_reason == "HALT"


def test_mov_opcode(vm: VM) -> None:
    program = [
        Instruction("CONST", dst=0, imm=11),
        Instruction("MOV", dst=1, src1=0),
        Instruction("HALT"),
    ]

    state = vm.execute(program)

    assert state.registers[1] == 11


@pytest.mark.parametrize(
    ("opcode", "lhs", "rhs", "expected"),
    [
        ("AND", True, False, False),
        ("OR", True, False, True),
        ("XOR", True, False, True),
    ],
)
def test_boolean_binary_opcodes(
    vm: VM,
    opcode: str,
    lhs: bool,
    rhs: bool,
    expected: bool,
) -> None:
    program = [
        Instruction("CONST", dst=0, imm=lhs, type_tag="bool"),
        Instruction("CONST", dst=1, imm=rhs, type_tag="bool"),
        Instruction(opcode, dst=2, src1=0, src2=1),
        Instruction("HALT"),
    ]

    state = vm.execute(program)

    assert state.registers[2] is expected


def test_not_opcode(vm: VM) -> None:
    program = [
        Instruction("CONST", dst=0, imm=True, type_tag="bool"),
        Instruction("NOT", dst=1, src1=0),
        Instruction("HALT"),
    ]

    state = vm.execute(program)

    assert state.registers[1] is False


@pytest.mark.parametrize(
    ("opcode", "lhs", "rhs", "expected", "flag"),
    [
        ("CMP_EQ", 5, 5, True, "eq"),
        ("CMP_LT", 2, 5, True, "lt"),
        ("CMP_GT", 7, 3, True, "gt"),
    ],
)
def test_comparison_opcodes_set_destination_and_flags(
    vm: VM,
    opcode: str,
    lhs: int,
    rhs: int,
    expected: bool,
    flag: str,
) -> None:
    program = [
        Instruction("CONST", dst=0, imm=lhs),
        Instruction("CONST", dst=1, imm=rhs),
        Instruction(opcode, dst=2, src1=0, src2=1),
        Instruction("HALT"),
    ]

    state = vm.execute(program)

    assert state.registers[2] is expected
    assert state.flags[flag] is True


def test_jmp_opcode(vm: VM) -> None:
    program = [
        Instruction("JMP", imm=2),
        Instruction("CONST", dst=0, imm=1),
        Instruction("CONST", dst=0, imm=2),
        Instruction("HALT"),
    ]

    state = vm.execute(program)

    assert state.registers[0] == 2


def test_jz_opcode_takes_branch_on_zero(vm: VM) -> None:
    program = [
        Instruction("CONST", dst=0, imm=0),
        Instruction("JZ", src1=0, imm=4),
        Instruction("CONST", dst=1, imm=99),
        Instruction("JMP", imm=5),
        Instruction("CONST", dst=1, imm=42),
        Instruction("HALT"),
    ]

    state = vm.execute(program)

    assert state.registers[1] == 42


def test_jnz_opcode_takes_branch_on_non_zero(vm: VM) -> None:
    program = [
        Instruction("CONST", dst=0, imm=1),
        Instruction("JNZ", src1=0, imm=4),
        Instruction("CONST", dst=1, imm=99),
        Instruction("JMP", imm=5),
        Instruction("CONST", dst=1, imm=42),
        Instruction("HALT"),
    ]

    state = vm.execute(program)

    assert state.registers[1] == 42


def test_jlt_opcode_uses_flags(vm: VM) -> None:
    program = [
        Instruction("CONST", dst=0, imm=2),
        Instruction("CONST", dst=1, imm=5),
        Instruction("CMP_LT", dst=2, src1=0, src2=1),
        Instruction("JLT", imm=6),
        Instruction("CONST", dst=3, imm=0),
        Instruction("JMP", imm=7),
        Instruction("CONST", dst=3, imm=1),
        Instruction("HALT"),
    ]

    state = vm.execute(program)

    assert state.registers[3] == 1


def test_jgt_opcode_uses_flags(vm: VM) -> None:
    program = [
        Instruction("CONST", dst=0, imm=7),
        Instruction("CONST", dst=1, imm=5),
        Instruction("CMP_GT", dst=2, src1=0, src2=1),
        Instruction("JGT", imm=6),
        Instruction("CONST", dst=3, imm=0),
        Instruction("JMP", imm=7),
        Instruction("CONST", dst=3, imm=1),
        Instruction("HALT"),
    ]

    state = vm.execute(program)

    assert state.registers[3] == 1


def test_load_and_store(vm: VM) -> None:
    program = [
        Instruction("CONST", dst=0, imm=10),
        Instruction("CONST", dst=1, imm=7),
        Instruction("STORE", src1=0, src2=1, imm=2),
        Instruction("LOAD", dst=2, src1=0, imm=2),
        Instruction("HALT"),
    ]

    state = vm.execute(program)

    assert state.memory[12] == 7
    assert state.registers[2] == 7


def test_push_and_pop(vm: VM) -> None:
    program = [
        Instruction("CONST", dst=0, imm=9),
        Instruction("PUSH", src1=0),
        Instruction("POP", dst=1),
        Instruction("HALT"),
    ]

    state = vm.execute(program)

    assert state.registers[1] == 9
    assert state.stack == []


def test_call_and_ret(vm: VM) -> None:
    program = [
        Instruction("CALL", imm=3),
        Instruction("HALT"),
        Instruction("HALT"),
        Instruction("CONST", dst=0, imm=13),
        Instruction("RET"),
    ]

    state = vm.execute(program)

    assert state.registers[0] == 13
    assert state.call_stack == []
    assert state.halt_reason == "HALT"


def test_timeout_trap() -> None:
    vm = VM(step_budget=3)
    program = [Instruction("JMP", imm=0)]

    with pytest.raises(Trap, match="TIMEOUT"):
        vm.execute(program)


def test_invalid_opcode_trap(vm: VM) -> None:
    with pytest.raises(Trap, match="INVALID_OP"):
        vm.execute([Instruction("BOGUS")])


def test_divide_by_zero_trap(vm: VM) -> None:
    program = [
        Instruction("CONST", dst=0, imm=6),
        Instruction("CONST", dst=1, imm=0),
        Instruction("DIV", dst=2, src1=0, src2=1),
    ]

    with pytest.raises(Trap, match="DIV0"):
        vm.execute(program)


def test_invalid_memory_address_trap(vm: VM) -> None:
    program = [
        Instruction("CONST", dst=0, imm=70_000),
        Instruction("LOAD", dst=1, src1=0),
    ]

    with pytest.raises(Trap, match="ADDR"):
        vm.execute(program)


def test_stack_underflow_trap(vm: VM) -> None:
    with pytest.raises(Trap, match="ADDR"):
        vm.execute([Instruction("POP", dst=0)])


def test_return_underflow_trap(vm: VM) -> None:
    with pytest.raises(Trap, match="ADDR"):
        vm.execute([Instruction("RET")])


def test_type_trap_for_invalid_add(vm: VM) -> None:
    program = [
        Instruction("CONST", dst=0, imm=True, type_tag="bool"),
        Instruction("CONST", dst=1, imm=3),
        Instruction("ADD", dst=2, src1=0, src2=1),
    ]

    with pytest.raises(Trap, match="TYPE"):
        vm.execute(program)


def test_checked_i32_overflow_trap(vm: VM) -> None:
    program = [
        Instruction("CONST", dst=0, imm=2**31 - 1, type_tag="checked_i32"),
        Instruction("CONST", dst=1, imm=1, type_tag="checked_i32"),
        Instruction("ADD", dst=2, src1=0, src2=1, type_tag="checked_i32"),
    ]

    with pytest.raises(Trap, match="OVERFLOW"):
        vm.execute(program)


def test_i32_wraparound(vm: VM) -> None:
    program = [
        Instruction("CONST", dst=0, imm=2**31 - 1, type_tag="i32"),
        Instruction("CONST", dst=1, imm=1, type_tag="i32"),
        Instruction("ADD", dst=2, src1=0, src2=1, type_tag="i32"),
        Instruction("HALT"),
    ]

    state = vm.execute(program)

    assert state.registers[2] == -(2**31)


def test_trace_capture(vm: VM) -> None:
    program = [
        Instruction("CONST", dst=0, imm=2),
        Instruction("CONST", dst=1, imm=3),
        Instruction("ADD", dst=2, src1=0, src2=1),
        Instruction("HALT"),
    ]

    state = vm.execute(program, trace=True)

    assert len(state.trace) == 4
    assert state.trace[0].instruction.opcode == "CONST"
    assert state.trace[-1].instruction.opcode == "HALT"
    assert state.trace[2].post_state["registers"][2] == 5


def test_execution_is_deterministic(vm: VM) -> None:
    program = [
        Instruction("CONST", dst=0, imm=2),
        Instruction("CONST", dst=1, imm=3),
        Instruction("ADD", dst=2, src1=0, src2=1),
        Instruction("HALT"),
    ]

    first = vm.execute(program, state=VMState(), trace=True)
    second = vm.execute(program, state=VMState(), trace=True)

    assert first.snapshot() == second.snapshot()
    assert [asdict(entry) for entry in first.trace] == [asdict(entry) for entry in second.trace]
