from __future__ import annotations

from types import MethodType
from typing import Any, cast

import pytest
import torch

from tesseract.backbone import (
    LearnedBackbone,
    RuleBasedBackbone,
    build_backbone_training_batch,
    build_learned_backbone,
    generate_nl_tasks,
    train_backbone_step,
)
from tesseract.compiler import build_training_batch, build_vocabularies, train_step
from tesseract.compiler.nl import BackboneConditionedCompiler
from tesseract.critic import DifferentialCritic
from tesseract.vm import Instruction, VM, ValidationError, validate_program


def _build_trained_nl_compiler() -> BackboneConditionedCompiler:
    nl_tasks = generate_nl_tasks(
        task_types=("arithmetic", "max", "sum_to_n"),
        operations=("add", "sub"),
        values=(1, 2, 3),
        seed=0,
    )
    synthetic_tasks = [task.to_synthetic_task() for task in nl_tasks]
    artifacts = build_vocabularies(synthetic_tasks)
    batch = build_training_batch(
        synthetic_tasks,
        prompt_vocab=artifacts.prompt_vocab,
        program_tokenizer=artifacts.program_tokenizer,
    )
    train_step(artifacts.compiler.model, batch)
    return BackboneConditionedCompiler(backbone=RuleBasedBackbone(), compiler=artifacts.compiler)


def _build_trained_learned_backbone() -> LearnedBackbone:
    nl_tasks = generate_nl_tasks(
        task_types=("arithmetic", "max", "sum_to_n"),
        operations=("add", "sub"),
        values=(1, 2, 3),
        seed=0,
        include_all_prompt_variants=True,
    )
    backbone = build_learned_backbone(nl_tasks)
    batch = build_backbone_training_batch(nl_tasks, canonical_vocabulary=backbone.model.canonical_vocabulary)
    train_backbone_step(backbone.model, batch)
    return backbone


def _build_trained_learned_nl_compiler() -> BackboneConditionedCompiler:
    nl_tasks = generate_nl_tasks(
        task_types=("arithmetic", "max", "sum_to_n"),
        operations=("add", "sub"),
        values=(1, 2, 3),
        seed=0,
        include_all_prompt_variants=True,
    )
    backbone = _build_trained_learned_backbone()
    synthetic_tasks = [task.to_synthetic_task() for task in nl_tasks]
    conditioning_vectors = [list(backbone.encode(task.prompt).conditioning) for task in nl_tasks]
    artifacts = build_vocabularies(synthetic_tasks)
    batch = build_training_batch(
        synthetic_tasks,
        prompt_vocab=artifacts.prompt_vocab,
        program_tokenizer=artifacts.program_tokenizer,
        conditioning_vectors=conditioning_vectors,
    )
    train_step(artifacts.compiler.model, batch)
    return BackboneConditionedCompiler(backbone=backbone, compiler=artifacts.compiler)


def test_rule_based_backbone_encodes_supported_prompts() -> None:
    tasks = generate_nl_tasks(task_types=("arithmetic", "max", "sum_to_n"), operations=("add",), values=(1, 2), seed=0)
    backbone = RuleBasedBackbone()

    outputs = [backbone.encode(task.prompt) for task in tasks]

    assert outputs
    assert {output.task_type for output in outputs} == {task.task_type for task in tasks}
    assert {output.canonical_prompt for output in outputs} == {task.canonical_prompt for task in tasks}


@pytest.mark.parametrize(
    ("prompt", "task_type", "canonical_prompt"),
    [
        ("What is 2 plus 3?", "arithmetic", "arith add 2 3"),
        ("Add 2 and 3", "arithmetic", "arith add 2 3"),
        ("What is 5 minus 2?", "arithmetic", "arith sub 5 2"),
        ("Subtract 2 from 5", "arithmetic", "arith sub 5 2"),
        ("What is 4 times 3?", "arithmetic", "arith mul 4 3"),
        ("Multiply 4 and 3", "arithmetic", "arith mul 4 3"),
        ("What is 8 divided by 2?", "arithmetic", "arith div 8 2"),
        ("Divide 8 by 2", "arithmetic", "arith div 8 2"),
        ("Max of 2 and 7", "max", "max 2 7"),
        ("Which number is larger: 2 or 7?", "max", "max 2 7"),
        ("Which is bigger, 2 or 7?", "max", "max 2 7"),
        ("Compare 2 and 7 and return the larger", "max", "max 2 7"),
        ("Sum integers from 1 to 4", "sum_to_n", "sum_to_n 4"),
        ("Sum numbers from 1 to 4", "sum_to_n", "sum_to_n 4"),
        ("Sum all integers up to 4", "sum_to_n", "sum_to_n 4"),
        ("Compute the triangular number of 4", "sum_to_n", "sum_to_n 4"),
    ],
)
def test_rule_based_backbone_covers_all_supported_prompt_variants(
    prompt: str,
    task_type: str,
    canonical_prompt: str,
) -> None:
    output = RuleBasedBackbone().encode(prompt)

    assert output.task_type == task_type
    assert output.canonical_prompt == canonical_prompt


def test_learned_backbone_overfits_scoped_nl_tasks() -> None:
    backbone = _build_trained_learned_backbone()
    tasks = generate_nl_tasks(
        task_types=("arithmetic", "max", "sum_to_n"),
        operations=("add", "sub"),
        values=(1, 2, 3),
        seed=0,
    )

    outputs = [backbone.encode(task.prompt) for task in tasks]

    assert {output.canonical_prompt for output in outputs} == {task.canonical_prompt for task in tasks}
    assert all(len(output.conditioning) == backbone.model.conditioning_dim for output in outputs)


def test_learned_backbone_training_is_seed_reproducible() -> None:
    tasks = generate_nl_tasks(
        task_types=("arithmetic", "max", "sum_to_n"),
        operations=("add", "sub"),
        values=(1, 2),
        seed=0,
    )

    torch.manual_seed(123)
    first = build_learned_backbone(tasks)
    first_batch = build_backbone_training_batch(tasks, canonical_vocabulary=first.model.canonical_vocabulary)
    train_backbone_step(first.model, first_batch, epochs=64)
    first_prompts = [first.encode(task.prompt).canonical_prompt for task in tasks]

    torch.manual_seed(123)
    second = build_learned_backbone(tasks)
    second_batch = build_backbone_training_batch(tasks, canonical_vocabulary=second.model.canonical_vocabulary)
    train_backbone_step(second.model, second_batch, epochs=64)
    second_prompts = [second.encode(task.prompt).canonical_prompt for task in tasks]

    assert first_prompts == second_prompts


def test_backbone_conditioned_compiler_runs_end_to_end_on_nl_tasks() -> None:
    compiler = _build_trained_nl_compiler()
    vm = VM()
    tasks = generate_nl_tasks(
        task_types=("arithmetic", "max", "sum_to_n"),
        operations=("add", "sub"),
        values=(1, 2),
        seed=0,
    )

    for task in tasks:
        result = compiler.execute(task.prompt, vm=vm)
        validate_program(result.program)
        assert result.output == task.expected_output
        assert result.backbone_output.canonical_prompt == task.canonical_prompt


def test_learned_backbone_conditioned_compiler_runs_end_to_end_on_nl_tasks() -> None:
    compiler = _build_trained_learned_nl_compiler()
    assert isinstance(compiler.backbone, LearnedBackbone)
    vm = VM()
    tasks = generate_nl_tasks(
        task_types=("arithmetic", "max", "sum_to_n"),
        operations=("add", "sub"),
        values=(1, 2),
        seed=0,
    )

    for task in tasks:
        result = compiler.execute(task.prompt, vm=vm)
        validate_program(result.program)
        assert result.output == task.expected_output
        assert result.backbone_output.canonical_prompt == task.canonical_prompt
        assert len(result.backbone_output.conditioning) == compiler.backbone.model.conditioning_dim


def test_backbone_conditioned_compiler_passes_learned_conditioning_to_compiler() -> None:
    compiler = _build_trained_learned_nl_compiler()
    captured: dict[str, object] = {}
    original_compile_conditioned = compiler.compiler.compile_conditioned

    def recording_compile_conditioned(prompt: str, conditioning=None):
        captured["prompt"] = prompt
        captured["conditioning"] = conditioning
        return original_compile_conditioned(prompt, conditioning)

    cast(Any, compiler.compiler).compile_conditioned = recording_compile_conditioned
    result = compiler.compile_with_backbone_output("What is 1 plus 2?")

    assert captured["prompt"] == result.backbone_output.canonical_prompt
    assert captured["conditioning"] == result.backbone_output.conditioning
    assert result.backbone_output.conditioning


def test_nl_pipeline_depends_on_emitted_ir_execution() -> None:
    compiler = _build_trained_nl_compiler()
    task = generate_nl_tasks(task_types=("arithmetic",), operations=("add",), values=(2,), seed=0)[0]

    good_result = compiler.execute(task.prompt)
    assert good_result.output == task.expected_output

    def wrong_compile(self: Any, prompt: str):
        del prompt
        return (Instruction("CONST", dst=2, imm=0), Instruction("HALT"))

    cast(Any, compiler.compiler).compile = MethodType(wrong_compile, compiler.compiler)
    bad_result = compiler.execute(task.prompt)

    assert bad_result.output != task.expected_output


def test_backbone_conditioned_compiler_rejects_unsupported_prompt() -> None:
    compiler = _build_trained_nl_compiler()

    with pytest.raises(ValueError, match="unsupported natural-language prompt"):
        compiler.compile("Explain recursion")


def test_generate_nl_tasks_rejects_unknown_task_types_and_operations() -> None:
    with pytest.raises(ValueError, match="unsupported task type"):
        generate_nl_tasks(task_types=("arithmetic", "unknown"))
    with pytest.raises(ValueError, match="unsupported operation"):
        generate_nl_tasks(task_types=("arithmetic",), operations=("add", "pow"))


def test_backbone_conditioned_compiler_threads_repair_hint_through_metadata() -> None:
    compiler = _build_trained_nl_compiler()
    critic = DifferentialCritic()
    baseline_state = VM().execute([Instruction("HALT")], trace=True)
    report = critic.compare(baseline_state, baseline_state, task_prompt="What is 1 plus 2?")

    compile_result = compiler.compile_with_backbone_output("What is 1 plus 2?", repair_context=report)

    assert compile_result.backbone_output.metadata["repair_hint"] == report.repair_prompt


def test_nl_critic_pipeline_can_compare_candidate_to_gold_program() -> None:
    compiler = _build_trained_nl_compiler()
    critic = DifferentialCritic()
    vm = VM()
    task = generate_nl_tasks(task_types=("max",), values=(2, 3), seed=0)[0]

    candidate = tuple(compiler.compile(task.prompt))
    report = critic.compare_programs(vm, candidate, task.gold_program, task_prompt=task.prompt)

    assert report.failure_type == "SUCCESS"
    assert report.status == "success"


def test_invalid_ir_ablation_breaks_nl_execution_accuracy() -> None:
    compiler = _build_trained_nl_compiler()
    tasks = generate_nl_tasks(task_types=("arithmetic",), operations=("add",), values=(1, 2), seed=0)

    def invalid_compile(self: Any, prompt: str):
        del prompt
        return ()

    cast(Any, compiler.compiler).compile = MethodType(invalid_compile, compiler.compiler)

    for task in tasks:
        compile_result = compiler.compile_with_backbone_output(task.prompt)
        assert compile_result.program == ()
        with pytest.raises(ValidationError, match="program must not be empty"):
            compiler.execute(task.prompt)
