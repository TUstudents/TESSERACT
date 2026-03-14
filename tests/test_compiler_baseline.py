from __future__ import annotations

import math
from dataclasses import replace
from types import MethodType
from typing import Any, cast

import pytest

from tesseract.compiler import (
    AutoregressiveCompiler,
    ProgramTokenizer,
    build_training_batch,
    build_vocabularies,
    evaluate_compiler,
    execute_task,
    generate_synthetic_tasks,
    train_step,
)
from tesseract.compiler.synthetic import (
    build_sum_to_n_program,
    evaluate_operation,
    make_synthetic_task,
)
from tesseract.vm import Instruction, validate_program


def test_generate_synthetic_tasks_have_valid_gold_programs() -> None:
    tasks = generate_synthetic_tasks(task_types=("arithmetic", "max", "sum_to_n"), values=range(1, 4))

    assert tasks
    program_lengths = {len(task.gold_program) for task in tasks}
    assert len(program_lengths) >= 2
    for task in tasks:
        validate_program(task.gold_program)
        result = execute_task(task)
        assert result.output == task.expected_output
        assert result.program_length == len(task.gold_program)


def test_training_step_returns_finite_loss() -> None:
    tasks = generate_synthetic_tasks(task_types=("arithmetic", "max"), operations=("add", "sub"), values=range(0, 3))
    artifacts = build_vocabularies(tasks)
    batch = build_training_batch(
        tasks,
        prompt_vocab=artifacts.prompt_vocab,
        program_tokenizer=artifacts.program_tokenizer,
    )

    metrics = train_step(artifacts.compiler.model, batch)

    assert math.isfinite(metrics["loss"])
    assert metrics["loss"] >= 0.0
    assert math.isfinite(metrics["sequence_error_rate"])


@pytest.fixture(scope="module")
def trained_compiler() -> tuple[AutoregressiveCompiler, list]:
    tasks = generate_synthetic_tasks(task_types=("arithmetic", "max", "sum_to_n"), operations=("add", "sub"), values=range(0, 4))
    artifacts = build_vocabularies(tasks)
    batch = build_training_batch(
        tasks,
        prompt_vocab=artifacts.prompt_vocab,
        program_tokenizer=artifacts.program_tokenizer,
    )

    for _ in range(2):
        train_step(artifacts.compiler.model, batch)

    return artifacts.compiler, tasks


def test_autoregressive_compiler_overfits_tiny_dataset(
    trained_compiler: tuple[AutoregressiveCompiler, list],
) -> None:
    compiler, tasks = trained_compiler
    metrics = evaluate_compiler(compiler, tasks)

    assert metrics.exact_output_accuracy == pytest.approx(1.0)
    assert metrics.exact_program_match == pytest.approx(1.0)
    assert metrics.compile_validity_rate == pytest.approx(1.0)
    assert metrics.execution_success_rate == pytest.approx(1.0)
    assert metrics.trap_rate == pytest.approx(0.0)
    assert metrics.average_program_length > 4.0


def test_compiler_compile_produces_valid_programs(
    trained_compiler: tuple[AutoregressiveCompiler, list],
) -> None:
    compiler, tasks = trained_compiler

    for task in tasks[:10]:
        program = tuple(compiler.compile(task.prompt))
        validate_program(program)
        assert program == task.gold_program


def test_compiler_execution_agrees_on_fixed_mini_benchmark(
    trained_compiler: tuple[AutoregressiveCompiler, list],
) -> None:
    compiler, _ = trained_compiler
    benchmark = generate_synthetic_tasks(task_types=("arithmetic", "sum_to_n"), operations=("add",), values=(0, 1, 2))
    metrics = evaluate_compiler(compiler, benchmark)

    assert metrics.exact_output_accuracy == pytest.approx(1.0)
    assert metrics.exact_program_match == pytest.approx(1.0)


def test_prompt_vocabulary_encodes_batch() -> None:
    tasks = generate_synthetic_tasks(task_types=("arithmetic",), operations=("add",), values=(0, 1))
    artifacts = build_vocabularies(tasks)

    encoded = artifacts.prompt_vocab.encode_batch([task.prompt for task in tasks])

    assert len(encoded) == len(tasks)
    assert all(len(item) >= 3 for item in encoded)


def test_program_tokenizer_round_trip() -> None:
    program = build_sum_to_n_program(3)
    tasks = generate_synthetic_tasks(task_types=("sum_to_n",), values=(3,))
    tokenizer = ProgramTokenizer(build_vocabularies(tasks).program_vocab)

    encoded = tokenizer.encode_program(program)
    decoded = tokenizer.decode_tokens(encoded)

    assert tuple(replace(instruction, label=None) for instruction in decoded) == tuple(
        replace(instruction, label=None) for instruction in program
    )


def test_evaluation_metrics_are_bounded() -> None:
    tasks = generate_synthetic_tasks(task_types=("arithmetic",), operations=("add",), values=(0, 1, 2))
    artifacts = build_vocabularies(tasks)
    batch = build_training_batch(
        tasks,
        prompt_vocab=artifacts.prompt_vocab,
        program_tokenizer=artifacts.program_tokenizer,
    )
    train_step(artifacts.compiler.model, batch)
    metrics = evaluate_compiler(artifacts.compiler, tasks)

    assert 0.0 <= metrics.exact_output_accuracy <= 1.0
    assert 0.0 <= metrics.exact_program_match <= 1.0
    assert 0.0 <= metrics.compile_validity_rate <= 1.0
    assert 0.0 <= metrics.execution_success_rate <= 1.0
    assert 0.0 <= metrics.trap_rate <= 1.0
    assert metrics.average_program_length > 0.0


def test_division_semantics_match_vm_truncation() -> None:
    assert evaluate_operation("div", -7, 2) == -3
    assert evaluate_operation("div", 7, -2) == -3


def test_execute_task_uses_task_result_register() -> None:
    task = make_synthetic_task("add", 2, 3, result_register=5)

    result = execute_task(task)

    assert result.output == 5
    assert result.result_register == 5


def test_evaluate_compiler_uses_task_result_register() -> None:
    tasks = [make_synthetic_task("add", 2, 3, result_register=5)]
    artifacts = build_vocabularies(tasks)
    batch = build_training_batch(
        tasks,
        prompt_vocab=artifacts.prompt_vocab,
        program_tokenizer=artifacts.program_tokenizer,
    )
    train_step(artifacts.compiler.model, batch)

    metrics = evaluate_compiler(artifacts.compiler, tasks)

    assert metrics.exact_output_accuracy == pytest.approx(1.0)
    assert metrics.exact_program_match == pytest.approx(1.0)


def test_evaluate_compiler_marks_structurally_invalid_programs_invalid() -> None:
    tasks = [make_synthetic_task("add", 2, 3)]
    artifacts = build_vocabularies(tasks)

    def invalid_compile(self: AutoregressiveCompiler, prompt: str):
        del prompt
        return (Instruction("CONST", dst=0, imm=1),)

    cast(Any, artifacts.compiler).compile = MethodType(invalid_compile, artifacts.compiler)
    metrics = evaluate_compiler(artifacts.compiler, tasks)

    assert metrics.compile_validity_rate == pytest.approx(0.0)
    assert metrics.exact_program_match == pytest.approx(0.0)
