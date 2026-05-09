from __future__ import annotations

import math
from pathlib import Path
from types import MethodType
from typing import Any, cast

import pytest
import torch

from tesseract.compiler import (
    AutoregressiveCompiler,
    PromptVocabulary,
    ProgramTokenizer,
    SyntheticTask,
    TrainingBatch,
    build_training_batch,
    build_vocabularies,
    evaluate_compiler,
    execute_task,
    generate_synthetic_tasks,
    train_step,
)
from tesseract.compiler.synthetic import (
    build_factorial_program,
    build_fibonacci_program,
    build_max_program,
    build_sum_to_n_program,
    evaluate_factorial,
    evaluate_fibonacci,
    evaluate_operation,
    make_abs_task,
    make_factorial_task,
    make_fibonacci_task,
    make_memory_sum_task,
    make_sum_to_n_task,
    make_synthetic_task,
)
from tesseract.vm import Instruction, ValidationError, assemble, validate_program


def test_generate_synthetic_tasks_have_valid_gold_programs() -> None:
    tasks = generate_synthetic_tasks(
        task_types=("arithmetic", "max", "sum_to_n", "factorial", "fibonacci", "abs", "memory_sum"),
        values=range(1, 4),
    )

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


def test_training_step_is_seed_reproducible() -> None:
    tasks = generate_synthetic_tasks(task_types=("arithmetic",), operations=("add",), values=(0, 1))

    torch.manual_seed(123)
    first = build_vocabularies(tasks)
    first_batch = build_training_batch(
        tasks,
        prompt_vocab=first.prompt_vocab,
        program_tokenizer=first.program_tokenizer,
    )
    train_step(first.compiler.model, first_batch, epochs=32)
    first_predictions = [first.compiler.predict_token_ids(task.prompt) for task in tasks]

    torch.manual_seed(123)
    second = build_vocabularies(tasks)
    second_batch = build_training_batch(
        tasks,
        prompt_vocab=second.prompt_vocab,
        program_tokenizer=second.program_tokenizer,
    )
    train_step(second.compiler.model, second_batch, epochs=32)
    second_predictions = [second.compiler.predict_token_ids(task.prompt) for task in tasks]

    assert first_predictions == second_predictions


def test_training_step_rejects_malformed_batches() -> None:
    tasks = generate_synthetic_tasks(task_types=("arithmetic",), operations=("add",), values=(0,))
    artifacts = build_vocabularies(tasks)
    batch = build_training_batch(
        tasks,
        prompt_vocab=artifacts.prompt_vocab,
        program_tokenizer=artifacts.program_tokenizer,
    )

    with pytest.raises(ValueError, match="encoded_prompts"):
        train_step(
            artifacts.compiler.model,
            TrainingBatch(tasks=tasks, encoded_prompts=[], encoded_programs=batch.encoded_programs),
        )
    with pytest.raises(ValueError, match="encoded_programs"):
        train_step(
            artifacts.compiler.model,
            TrainingBatch(tasks=tasks, encoded_prompts=batch.encoded_prompts, encoded_programs=[]),
        )
    with pytest.raises(ValueError, match="conditioning_vectors"):
        train_step(
            artifacts.compiler.model,
            TrainingBatch(
                tasks=tasks,
                encoded_prompts=batch.encoded_prompts,
                encoded_programs=batch.encoded_programs,
                conditioning_vectors=[],
            ),
        )
    with pytest.raises(ValueError, match="encoded_prompts contains invalid token"):
        train_step(
            artifacts.compiler.model,
            TrainingBatch(
                tasks=tasks,
                encoded_prompts=[[artifacts.prompt_vocab.size]],
                encoded_programs=batch.encoded_programs,
            ),
        )
    with pytest.raises(ValueError, match="encoded_programs contains invalid token"):
        train_step(
            artifacts.compiler.model,
            TrainingBatch(
                tasks=tasks,
                encoded_prompts=batch.encoded_prompts,
                encoded_programs=[[artifacts.program_vocab.size]],
            ),
        )


@pytest.fixture(scope="module")
def trained_compiler() -> tuple[AutoregressiveCompiler, list]:
    tasks = generate_synthetic_tasks(
        task_types=("arithmetic", "max", "sum_to_n", "factorial", "fibonacci", "abs", "memory_sum"),
        operations=("add", "sub"),
        values=range(0, 3),
    )
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
    assert metrics.token_accuracy == pytest.approx(1.0)
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


def test_prompt_vocabulary_maps_unseen_tokens_to_unk() -> None:
    tasks = generate_synthetic_tasks(task_types=("arithmetic",), operations=("add",), values=(0, 1))
    artifacts = build_vocabularies(tasks)

    encoded = artifacts.prompt_vocab.encode("arith add 99 100")

    assert encoded[0] == artifacts.prompt_vocab.stoi["arith"]
    assert encoded[1] == artifacts.prompt_vocab.stoi["add"]
    assert encoded[2] == artifacts.prompt_vocab.unk_id
    assert encoded[3] == artifacts.prompt_vocab.unk_id


def test_prompt_vocabulary_keeps_reserved_tokens_unique() -> None:
    task = SyntheticTask(
        prompt="<pad> arith <unk>",
        expected_output=0,
        gold_program=(Instruction("HALT"),),
    )

    vocabulary = PromptVocabulary.from_tasks([task])

    assert vocabulary.itos[:2] == ["<pad>", "<unk>"]
    assert len(vocabulary.itos) == len(set(vocabulary.itos))
    assert vocabulary.pad_id == 0
    assert vocabulary.unk_id == 1


def test_program_tokenizer_round_trip() -> None:
    program = build_sum_to_n_program(3)
    tasks = generate_synthetic_tasks(task_types=("sum_to_n",), values=(3,))
    tokenizer = ProgramTokenizer(build_vocabularies(tasks).program_vocab)

    encoded = tokenizer.encode_program(program)
    decoded = tokenizer.decode_tokens(encoded)

    assert decoded == program


def test_program_tokenizer_round_trip_preserves_labels() -> None:
    program = tuple(
        assemble(
            [
                "CONST dst=0 imm=0",
                "JZ src1=0 label=done",
                "CONST dst=1 imm=1",
                "done:",
                "HALT",
            ]
        )
    )
    tasks = [
        SyntheticTask(
            prompt="label round trip",
            expected_output=0,
            gold_program=program,
        )
    ]
    tokenizer = ProgramTokenizer(build_vocabularies(tasks).program_vocab)

    encoded = tokenizer.encode_program(program)
    decoded = tokenizer.decode_tokens(encoded)

    assert decoded == program


def test_program_tokenizer_round_trip_supports_f32_and_addr_tags() -> None:
    program = (
        Instruction("CONST", dst=0, imm=12, type_tag="addr"),
        Instruction("CONST", dst=1, imm=1.5, type_tag="f32"),
        Instruction("STORE", src1=0, src2=1, type_tag="f32"),
        Instruction("HALT"),
    )
    tasks = [SyntheticTask(prompt="typed round trip", expected_output=0, gold_program=program)]
    tokenizer = ProgramTokenizer(build_vocabularies(tasks).program_vocab)

    encoded = tokenizer.encode_program(program)
    decoded = tokenizer.decode_tokens(encoded)

    assert decoded == program


def test_program_tokenizer_rejects_malformed_or_truncated_sequences() -> None:
    tasks = generate_synthetic_tasks(task_types=("arithmetic",), operations=("add",), values=(0, 1))
    tokenizer = ProgramTokenizer(build_vocabularies(tasks).program_vocab)
    vocabulary = tokenizer.vocabulary
    assert vocabulary is not None

    malformed = [
        vocabulary.bos_id,
        vocabulary.stoi["CONST"],
        vocabulary.stoi["ADD"],
        vocabulary.sep_id,
        vocabulary.eos_id,
    ]
    duplicate = [
        vocabulary.bos_id,
        vocabulary.stoi["CONST"],
        vocabulary.stoi["dst=0"],
        vocabulary.stoi["dst=1"],
        vocabulary.sep_id,
        vocabulary.eos_id,
    ]
    truncated = [
        vocabulary.bos_id,
        vocabulary.stoi["CONST"],
        vocabulary.stoi["dst=0"],
        vocabulary.eos_id,
    ]
    trailing = [
        vocabulary.bos_id,
        vocabulary.stoi["HALT"],
        vocabulary.sep_id,
        vocabulary.eos_id,
        vocabulary.stoi["HALT"],
    ]

    with pytest.raises(ValidationError, match="malformed operand token"):
        tokenizer.decode_tokens(malformed)
    with pytest.raises(ValidationError, match="duplicate operand"):
        tokenizer.decode_tokens(duplicate)
    with pytest.raises(ValidationError, match="truncated instruction"):
        tokenizer.decode_tokens(truncated)
    with pytest.raises(ValidationError, match="tokens after <eos>"):
        tokenizer.decode_tokens(trailing)


def test_program_tokenizer_rejects_encoding_unknown_tokens() -> None:
    tasks = generate_synthetic_tasks(task_types=("arithmetic",), operations=("add",), values=(0, 1))
    tokenizer = ProgramTokenizer(build_vocabularies(tasks).program_vocab)
    unseen_program = build_sum_to_n_program(3)

    with pytest.raises(ValidationError, match="unknown program token"):
        tokenizer.encode_program(unseen_program)


def test_control_flow_gold_programs_preserve_symbolic_labels() -> None:
    max_program = build_max_program(3, 2)
    sum_program = build_sum_to_n_program(3)

    assert any(instruction.label is not None for instruction in max_program if instruction.opcode in {"JGT", "JMP"})
    assert any(instruction.label is not None for instruction in sum_program if instruction.opcode in {"JGT", "JMP"})


def test_neural_compiler_checkpoint_round_trip_preserves_predictions(tmp_path: Path) -> None:
    tasks = generate_synthetic_tasks(task_types=("arithmetic",), operations=("add",), values=(0, 1, 2))
    artifacts = build_vocabularies(tasks)
    batch = build_training_batch(
        tasks,
        prompt_vocab=artifacts.prompt_vocab,
        program_tokenizer=artifacts.program_tokenizer,
    )
    train_step(artifacts.compiler.model, batch)
    checkpoint_path = tmp_path / "compiler.pt"

    artifacts.compiler.save_checkpoint(checkpoint_path)
    restored = AutoregressiveCompiler.load_checkpoint(checkpoint_path)

    assert [restored.predict_token_ids(task.prompt) for task in tasks] == [
        artifacts.compiler.predict_token_ids(task.prompt) for task in tasks
    ]
    assert [tuple(restored.compile(task.prompt)) for task in tasks] == [
        tuple(artifacts.compiler.compile(task.prompt)) for task in tasks
    ]


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
    assert 0.0 <= metrics.token_accuracy <= 1.0
    assert metrics.average_program_length > 0.0


def test_evaluate_compiler_handles_empty_task_lists() -> None:
    artifacts = build_vocabularies([])

    metrics = evaluate_compiler(artifacts.compiler, [])

    assert metrics.exact_output_accuracy == pytest.approx(0.0)
    assert metrics.exact_program_match == pytest.approx(0.0)
    assert metrics.compile_validity_rate == pytest.approx(0.0)
    assert metrics.execution_success_rate == pytest.approx(0.0)
    assert metrics.average_program_length == pytest.approx(0.0)
    assert metrics.trap_rate == pytest.approx(0.0)


def test_division_semantics_match_vm_truncation() -> None:
    assert evaluate_operation("div", -7, 2) == -3
    assert evaluate_operation("div", 7, -2) == -3


def test_extended_task_builders_execute_correctly() -> None:
    factorial_task = make_factorial_task(5)
    fibonacci_task = make_fibonacci_task(7)
    abs_task = make_abs_task(-9)
    memory_sum_task = make_memory_sum_task((3, 1, 4))

    assert execute_task(factorial_task).output == evaluate_factorial(5)
    assert execute_task(fibonacci_task).output == evaluate_fibonacci(7)
    assert execute_task(abs_task).output == 9
    assert execute_task(memory_sum_task).output == 8


@pytest.mark.parametrize(
    ("builder", "value", "message"),
    [
        (build_factorial_program, -1, "factorial requires a non-negative integer"),
        (build_fibonacci_program, -1, "fibonacci requires a non-negative integer"),
    ],
)
def test_extended_unary_builders_reject_negative_inputs(builder, value: int, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        builder(value)


def test_execute_task_uses_task_result_register() -> None:
    task = make_synthetic_task("add", 2, 3, result_register=5)

    result = execute_task(task)

    assert result.output == 5
    assert result.result_register == 5


def test_max_program_tie_returns_shared_value() -> None:
    program = build_max_program(3, 3)
    task = SyntheticTask(prompt="max 3 3", expected_output=3, gold_program=program, task_type="max", lhs=3, rhs=3)

    result = execute_task(task)

    assert result.output == 3


@pytest.mark.parametrize("result_register", [0, 1, 3, 4, 5])
def test_sum_to_n_program_supports_nondefault_result_registers(result_register: int) -> None:
    tasks = generate_synthetic_tasks(task_types=("sum_to_n",), values=(3,), result_register=result_register)

    assert len(tasks) == 1
    result = execute_task(tasks[0])

    assert result.output == 6
    assert result.result_register == result_register


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
    assert metrics.execution_success_rate == pytest.approx(0.0)
    assert metrics.exact_program_match == pytest.approx(0.0)


def test_compiler_handles_unseen_prompt_tokens_without_crashing() -> None:
    tasks = generate_synthetic_tasks(task_types=("arithmetic",), operations=("add",), values=(0, 1))
    artifacts = build_vocabularies(tasks)
    batch = build_training_batch(
        tasks,
        prompt_vocab=artifacts.prompt_vocab,
        program_tokenizer=artifacts.program_tokenizer,
    )
    train_step(artifacts.compiler.model, batch)

    program = tuple(artifacts.compiler.compile("arith add 99 100"))

    validate_program(program)


def test_compiler_returns_empty_program_on_invalid_decoded_token_sequence() -> None:
    tasks = generate_synthetic_tasks(task_types=("arithmetic",), operations=("add",), values=(0, 1))
    artifacts = build_vocabularies(tasks)
    batch = build_training_batch(
        tasks,
        prompt_vocab=artifacts.prompt_vocab,
        program_tokenizer=artifacts.program_tokenizer,
    )
    train_step(artifacts.compiler.model, batch)

    def invalid_decode(self: AutoregressiveCompiler, prompt: str, *, max_steps: int = 256) -> list[int]:
        del prompt, max_steps
        vocabulary = self.program_tokenizer.vocabulary
        assert vocabulary is not None
        return [vocabulary.bos_id, len(vocabulary.itos), vocabulary.eos_id]

    cast(Any, artifacts.compiler.model).decode = MethodType(invalid_decode, artifacts.compiler.model)

    assert tuple(artifacts.compiler.compile("arith add 0 1")) == ()


def test_generate_synthetic_tasks_keeps_truncating_division_examples() -> None:
    tasks = generate_synthetic_tasks(task_types=("arithmetic",), operations=("div",), values=(2, 3))
    matching = [task for task in tasks if task.operation == "div" and task.lhs == 2 and task.rhs == 3]

    assert len(matching) == 1
    assert execute_task(matching[0]).output == 0


def test_generate_synthetic_tasks_rejects_unknown_task_types() -> None:
    with pytest.raises(ValueError, match="unsupported task type"):
        generate_synthetic_tasks(task_types=("arithmetic", "unknown"))


def test_generate_synthetic_tasks_rejects_unknown_operations() -> None:
    with pytest.raises(ValueError, match="unsupported operation"):
        generate_synthetic_tasks(task_types=("arithmetic",), operations=("add", "pow"))


def test_sum_to_n_rejects_negative_inputs() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        build_sum_to_n_program(-1)
    with pytest.raises(ValueError, match="non-negative"):
        make_sum_to_n_task(-1)
    with pytest.raises(ValueError, match="non-negative"):
        generate_synthetic_tasks(task_types=("sum_to_n",), values=(-1, 0))


def test_sum_to_n_zero_is_a_documented_degenerate_case() -> None:
    task = make_sum_to_n_task(0)

    result = execute_task(task)

    assert result.output == 0
    assert len(task.gold_program) > 4
