from __future__ import annotations

from typing import cast

import pytest

from tesseract.critic import (
    DifferentialCritic,
    FinalMemoryInvariant,
    FinalRegisterInvariant,
    Invariant,
    MaxStepsInvariant,
    NoTrapInvariant,
    TraceStepInvariant,
    build_repair_prompt,
    evaluate_invariants,
    summarize_trace,
)
from tesseract.vm import Instruction, VM, VMState, assemble


def test_summarize_trace_success_and_failure() -> None:
    vm = VM()
    success_state = vm.execute([Instruction("HALT")], trace=True)
    failure_state = VMState()
    with pytest.raises(Exception):
        vm.execute([Instruction("DIV", dst=0, src1=0, src2=1)], state=failure_state, trace=True)

    success_summary = summarize_trace(success_state)
    failure_summary = summarize_trace(failure_state)

    assert success_summary.status == "success"
    assert success_summary.halt_reason == "HALT"
    assert failure_summary.status == "failure"
    assert failure_summary.trap == "DIV0"


def test_differential_critic_detects_wrong_register_and_first_failing_step() -> None:
    vm = VM()
    critic = DifferentialCritic()
    gold = [
        Instruction("CONST", dst=0, imm=7),
        Instruction("CONST", dst=1, imm=3),
        Instruction("ADD", dst=2, src1=0, src2=1),
        Instruction("HALT"),
    ]
    candidate = [
        Instruction("CONST", dst=0, imm=7),
        Instruction("CONST", dst=1, imm=3),
        Instruction("SUB", dst=2, src1=0, src2=1),
        Instruction("HALT"),
    ]

    report = critic.compare_programs(vm, candidate, gold)

    assert report.failure_type == "WRONG_REGISTER"
    assert report.first_failing_step == 2
    assert report.differing_registers == (2,)


def test_differential_critic_detects_wrong_branch() -> None:
    vm = VM()
    critic = DifferentialCritic()
    gold = assemble(
        [
            "CONST dst=0 imm=0",
            "JZ src1=0 label=branch",
            "CONST dst=1 imm=1",
            "JMP label=done",
            "branch:",
            "CONST dst=1 imm=2",
            "done:",
            "HALT",
        ]
    )
    candidate = assemble(
        [
            "CONST dst=0 imm=0",
            "JNZ src1=0 label=branch",
            "CONST dst=1 imm=1",
            "JMP label=done",
            "branch:",
            "CONST dst=1 imm=2",
            "done:",
            "HALT",
        ]
    )

    report = critic.compare_programs(vm, candidate, gold)

    assert report.failure_type == "WRONG_BRANCH"
    assert report.first_failing_step == 1


def test_differential_critic_detects_wrong_address() -> None:
    vm = VM()
    critic = DifferentialCritic()
    gold = [
        Instruction("CONST", dst=0, imm=10),
        Instruction("CONST", dst=1, imm=7),
        Instruction("STORE", src1=0, src2=1, imm=0),
        Instruction("HALT"),
    ]
    candidate = [
        Instruction("CONST", dst=0, imm=10),
        Instruction("CONST", dst=1, imm=7),
        Instruction("STORE", src1=0, src2=1, imm=1),
        Instruction("HALT"),
    ]

    report = critic.compare_programs(vm, candidate, gold)

    assert report.failure_type == "WRONG_ADDRESS"
    assert report.first_failing_step == 2
    assert report.differing_addresses == (10, 11)


def test_differential_critic_detects_timeout() -> None:
    vm = VM(step_budget=3)
    critic = DifferentialCritic()
    gold = [Instruction("HALT")]
    candidate = [Instruction("JMP", imm=0)]

    report = critic.compare_programs(vm, candidate, gold)

    assert report.failure_type == "TIMEOUT"
    assert report.candidate_summary.halt_reason == "TIMEOUT"


def test_differential_critic_treats_matching_traps_as_success() -> None:
    vm = VM()
    critic = DifferentialCritic()
    gold = [Instruction("DIV", dst=0, src1=0, src2=1)]
    candidate = [Instruction("DIV", dst=0, src1=0, src2=1)]

    report = critic.compare_programs(vm, candidate, gold)

    assert report.status == "success"
    assert report.failure_type == "SUCCESS"
    assert report.first_failing_step is None


def test_differential_critic_detects_type_error() -> None:
    vm = VM()
    critic = DifferentialCritic()
    gold = [
        Instruction("CONST", dst=0, imm=1),
        Instruction("MOV", dst=1, src1=0),
        Instruction("HALT"),
    ]
    candidate = [
        Instruction("CONST", dst=0, imm=1),
        Instruction("MOV", dst=1, src1=0, type_tag="bool"),
        Instruction("HALT"),
    ]

    report = critic.compare_programs(vm, candidate, gold)

    assert report.failure_type == "TYPE_ERROR"
    assert report.first_failing_step == 1


def test_invariant_instrumentation_reports_violations() -> None:
    vm = VM()
    state = vm.execute(
        [
            Instruction("CONST", dst=0, imm=4),
            Instruction("CONST", dst=1, imm=5),
            Instruction("STORE", src1=0, src2=1),
            Instruction("HALT"),
        ],
        trace=True,
    )

    invariants = cast(
        tuple[Invariant, ...],
        (
            NoTrapInvariant(),
            FinalRegisterInvariant(register=0, expected=4),
            FinalMemoryInvariant(address=4, expected=6),
            MaxStepsInvariant(max_steps=2),
            TraceStepInvariant(step=1, register=1, expected=9),
        ),
    )
    violations = evaluate_invariants(state, invariants)

    assert len(violations) == 3
    assert {violation.name for violation in violations} == {"final_memory", "max_steps", "trace_step_register"}


def test_critic_analyze_returns_dict_with_repair_prompt() -> None:
    vm = VM()
    critic = DifferentialCritic()
    gold_state = vm.execute([Instruction("HALT")], trace=True)
    candidate_state = vm.execute([Instruction("HALT")], trace=True)

    report = critic.compare(candidate_state, gold_state, task_prompt="add 1 2")
    payload = critic.analyze(candidate_state, gold_state)

    assert report.repair_prompt == build_repair_prompt("add 1 2", report)
    assert payload["failure_type"] == "SUCCESS"
    assert "candidate trace matches expected trace" in payload["message"]


def test_differential_critic_uses_invariant_failure_type_for_matching_trace() -> None:
    vm = VM()
    critic = DifferentialCritic()
    gold_state = vm.execute([Instruction("HALT")], trace=True)
    candidate_state = vm.execute([Instruction("HALT")], trace=True)

    report = critic.compare(
        candidate_state,
        gold_state,
        invariants=cast(tuple[Invariant, ...], (FinalRegisterInvariant(register=0, expected=1),)),
    )

    assert report.status == "failure"
    assert report.failure_type == "INVARIANT_VIOLATION"
    assert len(report.invariant_violations) == 1


def test_differential_critic_compares_candidate_and_gold_traces_directly() -> None:
    vm = VM()
    critic = DifferentialCritic()
    gold_state = vm.execute(
        [
            Instruction("CONST", dst=0, imm=2),
            Instruction("CONST", dst=1, imm=3),
            Instruction("ADD", dst=2, src1=0, src2=1),
            Instruction("HALT"),
        ],
        trace=True,
    )
    candidate_state = vm.execute(
        [
            Instruction("CONST", dst=0, imm=2),
            Instruction("CONST", dst=1, imm=3),
            Instruction("SUB", dst=2, src1=0, src2=1),
            Instruction("HALT"),
        ],
        trace=True,
    )

    report = critic.compare(candidate_state.trace, gold_state.trace)

    assert report.first_failing_step == 2
    assert report.failure_type == "WRONG_REGISTER"
    assert report.differing_registers == (2,)


def test_differential_critic_compares_trace_less_final_states() -> None:
    critic = DifferentialCritic()
    candidate_state = VMState(registers={0: 1}, halted=True, halt_reason="HALT")
    gold_state = VMState(registers={0: 2}, halted=True, halt_reason="HALT")

    report = critic.compare(candidate_state, gold_state)

    assert report.status == "failure"
    assert report.first_failing_step is None
    assert report.failure_type == "WRONG_REGISTER"
    assert report.differing_registers == (0,)


def test_differential_critic_rejects_invalid_trace_payloads() -> None:
    critic = DifferentialCritic()
    gold_state = VMState(halted=True, halt_reason="HALT")

    with pytest.raises(TypeError, match="TraceEntry"):
        critic.compare(["not-a-trace-entry"], gold_state)
