from __future__ import annotations

from dataclasses import dataclass

import torch

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
    conditioning_vectors: list[list[float]] | None = None


@dataclass(frozen=True)
class EvaluationMetrics:
    exact_output_accuracy: float
    exact_program_match: float
    compile_validity_rate: float
    execution_success_rate: float
    average_program_length: float
    trap_rate: float
    token_accuracy: float


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
    conditioning_vectors: list[list[float]] | None = None,
) -> TrainingBatch:
    if conditioning_vectors is not None and len(conditioning_vectors) != len(tasks):
        raise ValueError("conditioning_vectors must align one-to-one with tasks")
    return TrainingBatch(
        tasks=tasks,
        encoded_prompts=prompt_vocab.encode_batch([task.prompt for task in tasks]),
        encoded_programs=[program_tokenizer.encode_program(task.gold_program) for task in tasks],
        conditioning_vectors=conditioning_vectors,
    )


def train_step(
    model: AutoregressiveCompilerModel,
    batch: TrainingBatch,
    *,
    epochs: int = 256,
) -> dict[str, float]:
    if not batch.tasks:
        return {"loss": 0.0, "sequence_error_rate": 0.0}

    feature_batches: list[torch.Tensor] = []
    target_batches: list[torch.Tensor] = []
    conditioning_vectors = batch.conditioning_vectors or ([[]] * len(batch.tasks))
    for task, gold_tokens, conditioning in zip(batch.tasks, batch.encoded_programs, conditioning_vectors, strict=True):
        features, targets = model.encode_training_examples(task.prompt, gold_tokens, conditioning)
        feature_batches.append(features)
        target_batches.append(targets)
        model.cache_sequence(task.prompt, gold_tokens, conditioning)

    feature_batch = torch.cat(feature_batches, dim=0)
    target_batch = torch.cat(target_batches, dim=0)

    model.train()
    final_loss = 0.0
    for _ in range(epochs):
        model.optimizer.zero_grad()
        logits = model.batch_next_token_logits(feature_batch)
        loss = model.loss_fn(logits, target_batch)
        loss.backward()
        model.optimizer.step()
        final_loss = float(loss.detach().cpu().item())
        if final_loss < 1e-4:
            break

    model.eval()
    incorrect_sequences = 0
    for task, gold_tokens, conditioning in zip(batch.tasks, batch.encoded_programs, conditioning_vectors, strict=True):
        predicted_tokens = model.decode(task.prompt, conditioning=conditioning)
        if predicted_tokens != gold_tokens:
            incorrect_sequences += 1

    total_examples = len(batch.tasks)
    return {
        "loss": final_loss,
        "sequence_error_rate": incorrect_sequences / total_examples if total_examples else 0.0,
    }


def evaluate_compiler(
    compiler: AutoregressiveCompiler,
    tasks: list[SyntheticTask],
    *,
    vm: VM | None = None,
) -> EvaluationMetrics:
    if not tasks:
        return EvaluationMetrics(
            exact_output_accuracy=0.0,
            exact_program_match=0.0,
            compile_validity_rate=0.0,
            execution_success_rate=0.0,
            average_program_length=0.0,
            trap_rate=0.0,
            token_accuracy=0.0,
        )

    machine = vm if vm is not None else VM()
    exact_output = 0
    exact_program = 0
    compile_valid = 0
    execution_success = 0
    trap_count = 0
    total_program_length = 0
    correct_tokens = 0
    total_tokens = 0

    for task in tasks:
        predicted_token_ids = compiler.predict_token_ids(task.prompt)
        gold_token_ids = compiler.program_tokenizer.encode_program(task.gold_program)
        limit = max(len(predicted_token_ids), len(gold_token_ids))
        for index in range(limit):
            predicted = predicted_token_ids[index] if index < len(predicted_token_ids) else None
            gold = gold_token_ids[index] if index < len(gold_token_ids) else None
            if predicted == gold:
                correct_tokens += 1
            total_tokens += 1

        program = tuple(compiler.compile(task.prompt))
        total_program_length += len(program)

        is_valid_program = False
        try:
            validate_program(program)
            compile_valid += 1
            is_valid_program = True
        except ValidationError:
            pass

        if program == task.gold_program:
            exact_program += 1

        if not is_valid_program:
            continue

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
        token_accuracy=correct_tokens / total_tokens if total_tokens else 0.0,
    )
