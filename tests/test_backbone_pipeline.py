from __future__ import annotations

from types import MethodType
from typing import Any, cast

import pytest

from tesseract.backbone import RuleBasedBackbone, generate_nl_tasks
from tesseract.compiler import build_training_batch, build_vocabularies, train_step
from tesseract.compiler.nl import BackboneConditionedCompiler
from tesseract.critic import DifferentialCritic
from tesseract.vm import Instruction, VM, validate_program


def _build_trained_nl_compiler() -> BackboneConditionedCompiler:
    nl_tasks = generate_nl_tasks(task_types=("arithmetic", "max", "sum_to_n"), operations=("add", "sub"), values=(1, 2, 3))
    synthetic_tasks = [task.to_synthetic_task() for task in nl_tasks]
    artifacts = build_vocabularies(synthetic_tasks)
    batch = build_training_batch(
        synthetic_tasks,
        prompt_vocab=artifacts.prompt_vocab,
        program_tokenizer=artifacts.program_tokenizer,
    )
    train_step(artifacts.compiler.model, batch)
    return BackboneConditionedCompiler(backbone=RuleBasedBackbone(), compiler=artifacts.compiler)


def test_rule_based_backbone_encodes_supported_prompts() -> None:
    tasks = generate_nl_tasks(task_types=("arithmetic", "max", "sum_to_n"), operations=("add",), values=(1, 2))
    backbone = RuleBasedBackbone()

    outputs = [backbone.encode(task.prompt) for task in tasks]

    assert outputs
    assert {output.task_type for output in outputs} == {task.task_type for task in tasks}
    assert {output.canonical_prompt for output in outputs} == {task.canonical_prompt for task in tasks}


def test_backbone_conditioned_compiler_runs_end_to_end_on_nl_tasks() -> None:
    compiler = _build_trained_nl_compiler()
    vm = VM()
    tasks = generate_nl_tasks(task_types=("arithmetic", "max", "sum_to_n"), operations=("add", "sub"), values=(1, 2))

    for task in tasks:
        result = compiler.execute(task.prompt, vm=vm)
        validate_program(result.program)
        assert result.output == task.expected_output
        assert result.backbone_output.canonical_prompt == task.canonical_prompt


def test_nl_pipeline_depends_on_emitted_ir_execution() -> None:
    compiler = _build_trained_nl_compiler()
    task = generate_nl_tasks(task_types=("arithmetic",), operations=("add",), values=(2,))[0]

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


def test_nl_critic_pipeline_can_compare_candidate_to_gold_program() -> None:
    compiler = _build_trained_nl_compiler()
    critic = DifferentialCritic()
    vm = VM()
    task = generate_nl_tasks(task_types=("max",), values=(2, 3))[0]

    candidate = tuple(compiler.compile(task.prompt))
    report = critic.compare_programs(vm, candidate, task.gold_program, task_prompt=task.prompt)

    assert report.failure_type == "SUCCESS"
    assert report.status == "success"


def test_invalid_ir_ablation_breaks_nl_execution_accuracy() -> None:
    compiler = _build_trained_nl_compiler()
    tasks = generate_nl_tasks(task_types=("arithmetic",), operations=("add",), values=(1, 2))

    def invalid_compile(self: Any, prompt: str):
        del prompt
        return ()

    cast(Any, compiler.compiler).compile = MethodType(invalid_compile, compiler.compiler)

    for task in tasks:
        compile_result = compiler.compile_with_backbone_output(task.prompt)
        assert compile_result.program == ()
