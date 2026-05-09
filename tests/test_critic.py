from __future__ import annotations

from typing import Any, cast

import pytest

from tesseract.compiler.synthetic import make_max_task, make_sum_to_n_task, make_synthetic_task
from tesseract.critic import (
    CriticTrainingExample,
    DifferentialCritic,
    FinalMemoryInvariant,
    FinalRegisterInvariant,
    Invariant,
    LearnedCritic,
    MaxStepsInvariant,
    NoTrapInvariant,
    TraceStepInvariant,
    build_critic_training_examples,
    build_learned_critic,
    build_repair_prompt,
    evaluate_invariants,
    evaluate_learned_critic,
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
        critic.compare(cast(Any, ["not-a-trace-entry"]), gold_state)


def test_differential_critic_prioritizes_trap_halt_reason_for_trace_less_states() -> None:
    critic = DifferentialCritic()
    candidate_state = VMState(registers={0: 1}, halted=True, halt_reason="TIMEOUT")
    gold_state = VMState(registers={0: 0}, halted=True, halt_reason="HALT")

    report = critic.compare(candidate_state, gold_state)

    assert report.status == "failure"
    assert report.first_failing_step is None
    assert report.failure_type == "TIMEOUT"


def test_build_critic_training_examples_is_deterministic_and_mixed() -> None:
    tasks = [
        make_synthetic_task("add", 2, 3),
        make_max_task(3, 1),
        make_sum_to_n_task(3),
    ]

    first = build_critic_training_examples(tasks)
    second = build_critic_training_examples(tasks)

    assert first == second
    assert any(example.failure_type == "SUCCESS" for example in first)
    assert any(example.failure_type != "SUCCESS" for example in first)


def test_learned_critic_rejects_invalid_trace_payloads() -> None:
    critic = build_learned_critic()
    gold_state = VMState(halted=True, halt_reason="HALT")

    with pytest.raises(TypeError, match="TraceEntry"):
        critic.compare(cast(Any, ["not-a-trace-entry"]), gold_state)


def test_learned_critic_validates_training_examples() -> None:
    critic = build_learned_critic()
    report = DifferentialCritic().compare(
        VMState(halted=True, halt_reason="HALT"),
        VMState(halted=True, halt_reason="HALT"),
    )
    valid_features = (0.0,) * critic.model.input_dim

    with pytest.raises(ValueError, match="feature length"):
        critic.fit(
            [
                CriticTrainingExample(
                    features=(0.0,),
                    failure_type="SUCCESS",
                    first_failing_step=None,
                    oracle_report=report,
                )
            ],
            epochs=1,
        )

    with pytest.raises(ValueError, match="unknown failure_type"):
        critic.fit(
            [
                CriticTrainingExample(
                    features=valid_features,
                    failure_type=cast(Any, "NOPE"),
                    first_failing_step=None,
                    oracle_report=report,
                )
            ],
            epochs=1,
        )

    with pytest.raises(ValueError, match="negative first_failing_step"):
        critic.fit(
            [
                CriticTrainingExample(
                    features=valid_features,
                    failure_type="WRONG_REGISTER",
                    first_failing_step=-1,
                    oracle_report=report,
                )
            ],
            epochs=1,
        )


@pytest.fixture(scope="module")
def trained_learned_critic() -> tuple[LearnedCritic, list]:
    tasks = [
        make_synthetic_task("add", 2, 3),
        make_synthetic_task("sub", 5, 1),
        make_max_task(3, 1),
        make_sum_to_n_task(3),
    ]
    examples = build_critic_training_examples(tasks)
    critic = build_learned_critic()
    metrics = critic.fit(examples, epochs=256)

    assert metrics["failure_type_accuracy"] == pytest.approx(1.0)
    assert metrics["first_step_accuracy"] == pytest.approx(1.0)
    return critic, examples


def test_learned_critic_matches_oracle_labels_on_training_examples(
    trained_learned_critic: tuple[LearnedCritic, list],
) -> None:
    critic, examples = trained_learned_critic

    metrics = evaluate_learned_critic(critic, examples)

    assert metrics.failure_type_accuracy == pytest.approx(1.0)
    assert metrics.first_step_accuracy == pytest.approx(1.0)


def test_learned_critic_compare_returns_schema_compatible_report(
    trained_learned_critic: tuple[LearnedCritic, list],
) -> None:
    critic, examples = trained_learned_critic
    example = next(example for example in examples if example.failure_type != "SUCCESS")
    oracle_report = example.oracle_report

    report = critic.compare(
        VMState(
            registers=dict(oracle_report.candidate_summary.final_registers),
            memory=dict(oracle_report.candidate_summary.final_memory),
            halted=oracle_report.candidate_summary.halt_reason == "HALT",
            halt_reason=oracle_report.candidate_summary.halt_reason,
            step_count=oracle_report.candidate_summary.step_count,
            pc=oracle_report.candidate_summary.final_pc,
        ),
        VMState(
            registers=dict(oracle_report.expected_summary.final_registers) if oracle_report.expected_summary is not None else {},
            memory=dict(oracle_report.expected_summary.final_memory) if oracle_report.expected_summary is not None else {},
            halted=(oracle_report.expected_summary.halt_reason == "HALT") if oracle_report.expected_summary is not None else False,
            halt_reason=oracle_report.expected_summary.halt_reason if oracle_report.expected_summary is not None else None,
            step_count=oracle_report.expected_summary.step_count if oracle_report.expected_summary is not None else 0,
            pc=oracle_report.expected_summary.final_pc if oracle_report.expected_summary is not None else 0,
        ),
        task_prompt=example.task_prompt,
    )

    assert report.failure_type in {
        "SUCCESS",
        "WRONG_BRANCH",
        "WRONG_REGISTER",
        "WRONG_ADDRESS",
        "WRONG_VALUE",
        "TYPE_ERROR",
        "TIMEOUT",
        "INVALID_OP",
        "INVARIANT_VIOLATION",
        "UNKNOWN_FAILURE",
    }
    assert report.status in {"success", "failure"}
    assert report.message
    assert report.repair_prompt is not None
