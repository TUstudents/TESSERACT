from __future__ import annotations

from dataclasses import dataclass

from tesseract.compiler.baseline import (
    AutoregressiveCompiler,
    AutoregressiveCompilerModel,
    ProgramTokenizer,
    ProgramVocabulary,
    PromptVocabulary,
)
from tesseract.compiler.synthetic import SyntheticTask
from tesseract.vm import Trap, VM, ValidationError, validate_program


@dataclass
class TrainingBatch:
    tasks: list[SyntheticTask]
    encoded_prompts: list[list[int]]
    encoded_programs: list[list[int]]


@dataclass(frozen=True)
class EvaluationMetrics:
    exact_output_accuracy: float
    exact_program_match: float
    compile_validity_rate: float
    execution_success_rate: float
    average_program_length: float
    trap_rate: float


@dataclass(frozen=True)
class CompilerArtifacts:
    compiler: AutoregressiveCompiler
    prompt_vocab: PromptVocabulary
    program_vocab: ProgramVocabulary
    program_tokenizer: ProgramTokenizer


def build_vocabularies(tasks: list[SyntheticTask]) -> CompilerArtifacts:
    prompt_vocab = PromptVocabulary.from_tasks(tasks)
    program_vocab = ProgramVocabulary.from_tasks(tasks)
    program_tokenizer = ProgramTokenizer(program_vocab)
    model = AutoregressiveCompilerModel(
        prompt_vocab=prompt_vocab,
        program_tokenizer=program_tokenizer,
    )
    compiler = AutoregressiveCompiler(model=model, program_tokenizer=program_tokenizer)
    return CompilerArtifacts(
        compiler=compiler,
        prompt_vocab=prompt_vocab,
        program_vocab=program_vocab,
        program_tokenizer=program_tokenizer,
    )


def build_training_batch(
    tasks: list[SyntheticTask],
    *,
    prompt_vocab: PromptVocabulary,
    program_tokenizer: ProgramTokenizer,
) -> TrainingBatch:
    return TrainingBatch(
        tasks=tasks,
        encoded_prompts=prompt_vocab.encode_batch([task.prompt for task in tasks]),
        encoded_programs=[program_tokenizer.encode_program(task.gold_program) for task in tasks],
    )


def train_step(
    model: AutoregressiveCompilerModel,
    batch: TrainingBatch,
) -> dict[str, float]:
    incorrect_tokens = 0
    total_tokens = 0
    incorrect_sequences = 0

    for task, gold_tokens in zip(batch.tasks, batch.encoded_programs, strict=True):
        predicted_tokens = model.decode(task.prompt)
        sequence_match = predicted_tokens == gold_tokens
        if not sequence_match:
            incorrect_sequences += 1
        limit = max(len(predicted_tokens), len(gold_tokens))
        for index in range(limit):
            predicted = predicted_tokens[index] if index < len(predicted_tokens) else None
            gold = gold_tokens[index] if index < len(gold_tokens) else None
            if predicted != gold:
                incorrect_tokens += 1
            total_tokens += 1

    model.update(batch.tasks)

    total_examples = len(batch.tasks)
    return {
        "loss": incorrect_tokens / total_tokens if total_tokens else 0.0,
        "sequence_error_rate": incorrect_sequences / total_examples if total_examples else 0.0,
    }


def evaluate_compiler(
    compiler: AutoregressiveCompiler,
    tasks: list[SyntheticTask],
    *,
    vm: VM | None = None,
) -> EvaluationMetrics:
    machine = vm if vm is not None else VM()
    exact_output = 0
    exact_program = 0
    compile_valid = 0
    execution_success = 0
    trap_count = 0
    total_program_length = 0

    for task in tasks:
        program = tuple(compiler.compile(task.prompt))
        total_program_length += len(program)

        try:
            validate_program(program)
            compile_valid += 1
        except ValidationError:
            pass

        if program == task.gold_program:
            exact_program += 1

        try:
            state = machine.execute(program)
            execution_success += 1
            if state.registers.get(task.result_register) == task.expected_output:
                exact_output += 1
        except Trap:
            trap_count += 1

    total = len(tasks)
    return EvaluationMetrics(
        exact_output_accuracy=exact_output / total,
        exact_program_match=exact_program / total,
        compile_validity_rate=compile_valid / total,
        execution_success_rate=execution_success / total,
        average_program_length=total_program_length / total,
        trap_rate=trap_count / total,
    )
