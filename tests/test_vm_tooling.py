from __future__ import annotations

from dataclasses import replace

import pytest

from tesseract.vm import (
    Instruction,
    Trap,
    ValidationError,
    VM,
    assemble,
    disassemble,
    program_from_json,
    program_to_json,
    replay_program,
    state_from_json,
    state_to_json,
    trace_from_json,
    trace_to_json,
    trap_from_json,
    trap_to_json,
    validate_program,
)


def _normalize_labels(program: list[Instruction]) -> list[Instruction]:
    return [replace(instruction, label=None) for instruction in program]


def test_assemble_resolves_labels() -> None:
    source = [
        "start:",
        "CONST dst=0 imm=1",
        "JMP label=end",
        "CONST dst=0 imm=99",
        "end:",
        "HALT",
    ]

    program = assemble(source)

    assert program[1].opcode == "JMP"
    assert program[1].imm == 3
    assert program[1].label == "end"


def test_disassemble_round_trip() -> None:
    source = [
        "entry:",
        "CONST dst=0 imm=1",
        "JZ src1=0 label=Ldone",
        "CONST dst=1 imm=2",
        "Ldone:",
        "HALT",
    ]

    first = assemble(source)
    text = disassemble(first)
    second = assemble(text)

    assert _normalize_labels(first) == _normalize_labels(second)


def test_validate_program_rejects_invalid_register() -> None:
    with pytest.raises(ValidationError, match="out of range"):
        validate_program([Instruction("CONST", dst=99, imm=1)])


def test_validate_program_rejects_unresolved_label() -> None:
    with pytest.raises(ValidationError, match="unresolved label"):
        validate_program([Instruction("JMP", label="missing")])


def test_assemble_rejects_undefined_label() -> None:
    with pytest.raises(ValidationError, match="undefined label"):
        assemble(["JMP label=missing"])


def test_assemble_rejects_duplicate_label() -> None:
    with pytest.raises(ValidationError, match="duplicate label"):
        assemble(["loop:", "loop:", "HALT"])


def test_assemble_rejects_malformed_operand() -> None:
    with pytest.raises(ValidationError, match="malformed operand"):
        assemble(["CONST dst=0 imm"])


def test_validate_program_type_checks_bool_add() -> None:
    program = [
        Instruction("CONST", dst=0, imm=True, type_tag="bool"),
        Instruction("CONST", dst=1, imm=1),
        Instruction("ADD", dst=2, src1=0, src2=1),
    ]

    with pytest.raises(ValidationError, match="expected one of"):
        validate_program(program)


def test_program_json_round_trip() -> None:
    program = [
        Instruction("CONST", dst=0, imm=5),
        Instruction("HALT"),
    ]

    payload = program_to_json(program)
    restored = program_from_json(payload)

    assert restored == program


def test_trace_json_round_trip() -> None:
    vm = VM()
    program = [
        Instruction("CONST", dst=0, imm=2),
        Instruction("CONST", dst=1, imm=3),
        Instruction("ADD", dst=2, src1=0, src2=1),
        Instruction("HALT"),
    ]

    state = vm.execute(program, trace=True)
    payload = trace_to_json(state.trace)
    restored = trace_from_json(payload)

    assert restored == state.trace


def test_state_json_round_trip() -> None:
    vm = VM()
    program = [
        Instruction("CONST", dst=0, imm=4),
        Instruction("PUSH", src1=0),
        Instruction("HALT"),
    ]

    state = vm.execute(program, trace=True)
    payload = state_to_json(state)
    restored = state_from_json(payload)

    assert restored.snapshot() == state.snapshot()
    assert restored.trace == state.trace


def test_trap_json_round_trip() -> None:
    trap = Trap("INVALID_OP", pc=3, instruction=Instruction("BOGUS"))

    payload = trap_to_json(trap)
    restored = trap_from_json(payload)

    assert restored == trap


def test_replay_program_reproduces_execution() -> None:
    vm = VM()
    program = [
        Instruction("CONST", dst=0, imm=8),
        Instruction("CONST", dst=1, imm=5),
        Instruction("SUB", dst=2, src1=0, src2=1),
        Instruction("HALT"),
    ]

    baseline = vm.execute(program, trace=True)
    replayed = replay_program(program_to_json(program), trace=True)

    assert replayed.snapshot() == baseline.snapshot()
    assert replayed.trace == baseline.trace
