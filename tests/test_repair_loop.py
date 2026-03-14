from __future__ import annotations

from dataclasses import dataclass

import pytest

from tesseract.backbone import RuleBasedBackbone, generate_nl_tasks
from tesseract.compiler import build_training_batch, build_vocabularies, train_step
from tesseract.compiler.nl import BackboneConditionedCompiler, NaturalLanguageCompileResult
from tesseract.critic import DifferentialCritic, RepairLoopController, evaluate_repair_loop
from tesseract.vm import Instruction


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


@dataclass
class FailThenRepairCompiler:
    delegate: BackboneConditionedCompiler
    failed_once: bool = False

    def compile_with_backbone_output(self, prompt: str, *, repair_context=None) -> NaturalLanguageCompileResult:  # type: ignore[no-untyped-def]
        del repair_context
        if not self.failed_once:
            self.failed_once = True
            backbone_output = self.delegate.backbone.encode(prompt)
            return NaturalLanguageCompileResult(backbone_output=backbone_output, program=())
        return self.delegate.compile_with_backbone_output(prompt)

    def repair_compile(self, prompt: str, report) -> NaturalLanguageCompileResult:  # type: ignore[no-untyped-def]
        del report
        return self.delegate.compile_with_backbone_output(prompt)


@dataclass
class OscillatingCompiler:
    delegate: BackboneConditionedCompiler

    def compile_with_backbone_output(self, prompt: str, *, repair_context=None) -> NaturalLanguageCompileResult:  # type: ignore[no-untyped-def]
        del repair_context
        backbone_output = self.delegate.backbone.encode(prompt)
        return NaturalLanguageCompileResult(
            backbone_output=backbone_output,
            program=(Instruction("CONST", dst=0, imm=1),),
        )

    def repair_compile(self, prompt: str, report) -> NaturalLanguageCompileResult:  # type: ignore[no-untyped-def]
        del report
        return self.compile_with_backbone_output(prompt)


def test_repair_loop_improves_over_failed_first_attempt() -> None:
    compiler = _build_trained_nl_compiler()
    repair_compiler = FailThenRepairCompiler(delegate=compiler)
    task = generate_nl_tasks(task_types=("arithmetic",), operations=("add",), values=(2,))[0]
    controller = RepairLoopController(critic=DifferentialCritic(), max_rounds=3)

    result = controller.run(task, repair_compiler)  # type: ignore[arg-type]

    assert result.success is True
    assert result.termination_reason == "success"
    assert result.rounds_used == 2
    assert result.attempts[0].compile_result.program == ()
    assert result.final_program == task.gold_program


def test_repair_loop_detects_oscillation() -> None:
    compiler = _build_trained_nl_compiler()
    oscillating_compiler = OscillatingCompiler(delegate=compiler)
    task = generate_nl_tasks(task_types=("arithmetic",), operations=("add",), values=(2,))[0]
    controller = RepairLoopController(critic=DifferentialCritic(), max_rounds=3)

    result = controller.run(task, oscillating_compiler)  # type: ignore[arg-type]

    assert result.success is False
    assert result.termination_reason == "oscillation"
    assert result.rounds_used == 2


def test_repair_loop_metrics_capture_success_and_failure_rates() -> None:
    compiler = _build_trained_nl_compiler()
    task = generate_nl_tasks(task_types=("arithmetic",), operations=("add",), values=(2,))[0]
    controller = RepairLoopController(critic=DifferentialCritic(), max_rounds=3)

    success_result = controller.run(task, FailThenRepairCompiler(delegate=compiler))  # type: ignore[arg-type]
    failed_result = controller.run(task, OscillatingCompiler(delegate=compiler))  # type: ignore[arg-type]
    metrics = evaluate_repair_loop((success_result, failed_result))

    assert metrics.success_after_1_round == pytest.approx(0.0)
    assert metrics.success_after_2_rounds == pytest.approx(0.5)
    assert metrics.success_after_3_rounds == pytest.approx(0.5)
    assert metrics.non_convergence_rate == pytest.approx(0.5)
    assert metrics.oscillation_rate == pytest.approx(0.5)
    assert metrics.average_rounds == pytest.approx(2.0)
