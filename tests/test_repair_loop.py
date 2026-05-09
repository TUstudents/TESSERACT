from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import pytest

from tesseract.backbone import RuleBasedBackbone, generate_nl_tasks
from tesseract.compiler import build_training_batch, build_vocabularies, train_step
from tesseract.compiler.nl import BackboneConditionedCompiler, NaturalLanguageCompileResult
from tesseract.critic import (
    CriticReport,
    DifferentialCritic,
    RepairLoopController,
    RepairState,
    RepairTrainingExample,
    RepairVocabulary,
    build_held_out_repair_benchmark,
    build_model_driven_repair_compiler,
    build_repair_state,
    build_repair_training_examples,
    evaluate_model_driven_repair,
    evaluate_repair_loop,
    run_repair_benchmark,
)
from tesseract.critic.loop import build_repair_context
from tesseract.vm import Instruction, VM


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


@dataclass
class FailThenRepairCompiler:
    delegate: BackboneConditionedCompiler
    failed_once: bool = False

    def compile_with_backbone_output(
        self,
        prompt: str,
        *,
        repair_context: CriticReport | None = None,
    ) -> NaturalLanguageCompileResult:
        del repair_context
        if not self.failed_once:
            self.failed_once = True
            backbone_output = self.delegate.backbone.encode(prompt)
            return NaturalLanguageCompileResult(backbone_output=backbone_output, program=())
        return self.delegate.compile_with_backbone_output(prompt)

    def repair_compile(self, prompt: str, report: CriticReport) -> NaturalLanguageCompileResult:
        del report
        return self.delegate.compile_with_backbone_output(prompt)


@dataclass
class OscillatingCompiler:
    delegate: BackboneConditionedCompiler

    def compile_with_backbone_output(
        self,
        prompt: str,
        *,
        repair_context: CriticReport | None = None,
    ) -> NaturalLanguageCompileResult:
        del repair_context
        backbone_output = self.delegate.backbone.encode(prompt)
        return NaturalLanguageCompileResult(
            backbone_output=backbone_output,
            program=(Instruction("CONST", dst=0, imm=1),),
        )

    def repair_compile(self, prompt: str, report: CriticReport) -> NaturalLanguageCompileResult:
        del report
        return self.compile_with_backbone_output(prompt)


@dataclass
class UniqueInvalidCompiler:
    delegate: BackboneConditionedCompiler
    attempts: int = 0

    def compile_with_backbone_output(
        self,
        prompt: str,
        *,
        repair_context: CriticReport | None = None,
    ) -> NaturalLanguageCompileResult:
        del repair_context
        backbone_output = self.delegate.backbone.encode(prompt)
        program = (Instruction("CONST", dst=0, imm=self.attempts + 1),)
        self.attempts += 1
        return NaturalLanguageCompileResult(backbone_output=backbone_output, program=program)

    def repair_compile(self, prompt: str, report: CriticReport) -> NaturalLanguageCompileResult:
        del report
        return self.compile_with_backbone_output(prompt)


def test_repair_loop_improves_over_failed_first_attempt() -> None:
    compiler = _build_trained_nl_compiler()
    repair_compiler = FailThenRepairCompiler(delegate=compiler)
    task = generate_nl_tasks(task_types=("arithmetic",), operations=("add",), values=(2,), seed=0)[0]
    controller = RepairLoopController(critic=DifferentialCritic(), max_rounds=3)

    result = controller.run(task, repair_compiler)

    assert result.success is True
    assert result.termination_reason == "success"
    assert result.rounds_used == 2
    assert result.attempts[0].compile_result.program == ()
    assert result.attempts[0].repair_context is not None
    assert result.final_program == task.gold_program


def test_repair_loop_detects_oscillation() -> None:
    compiler = _build_trained_nl_compiler()
    oscillating_compiler = OscillatingCompiler(delegate=compiler)
    task = generate_nl_tasks(task_types=("arithmetic",), operations=("add",), values=(2,), seed=0)[0]
    controller = RepairLoopController(critic=DifferentialCritic(), max_rounds=3)

    result = controller.run(task, oscillating_compiler)

    assert result.success is False
    assert result.termination_reason == "oscillation"
    assert result.rounds_used == 2


def test_repair_loop_stops_at_max_rounds_for_non_oscillating_failures() -> None:
    compiler = _build_trained_nl_compiler()
    invalid_compiler = UniqueInvalidCompiler(delegate=compiler)
    task = generate_nl_tasks(task_types=("arithmetic",), operations=("add",), values=(2,), seed=0)[0]
    controller = RepairLoopController(critic=DifferentialCritic(), max_rounds=3)

    result = controller.run(task, invalid_compiler)

    assert result.success is False
    assert result.termination_reason == "max_rounds"
    assert result.rounds_used == 3


def test_build_repair_context_captures_failure_details() -> None:
    task = generate_nl_tasks(task_types=("arithmetic",), operations=("add",), values=(2,), seed=0)[0]
    critic = DifferentialCritic()
    vm = VM()
    report = critic.compare_programs(
        vm,
        (Instruction("CONST", dst=0, imm=1),),
        task.gold_program,
        task_prompt=task.prompt,
    )

    context = build_repair_context(
        task_prompt=task.prompt,
        candidate_program=(Instruction("CONST", dst=0, imm=1),),
        critic_report=report,
        round_index=0,
    )

    assert context.task_prompt == task.prompt
    assert context.candidate_program == (Instruction("CONST", dst=0, imm=1),)
    assert context.critic_report.first_failing_step is not None
    assert context.critic_report.repair_prompt is not None
    assert context.round_index == 0


def test_repair_loop_metrics_capture_success_and_failure_rates() -> None:
    compiler = _build_trained_nl_compiler()
    task = generate_nl_tasks(task_types=("arithmetic",), operations=("add",), values=(2,), seed=0)[0]
    controller = RepairLoopController(critic=DifferentialCritic(), max_rounds=3)

    success_result = controller.run(task, FailThenRepairCompiler(delegate=compiler))
    failed_result = controller.run(task, OscillatingCompiler(delegate=compiler))
    metrics = evaluate_repair_loop((success_result, failed_result))

    assert metrics.success_after_1_round == pytest.approx(0.0)
    assert metrics.success_after_2_rounds == pytest.approx(0.5)
    assert metrics.success_after_3_rounds == pytest.approx(0.5)
    assert metrics.non_convergence_rate == pytest.approx(0.5)
    assert metrics.oscillation_rate == pytest.approx(0.5)
    assert metrics.average_rounds == pytest.approx(2.0)
    assert metrics.average_extra_steps == pytest.approx(1.0)


def test_repair_state_round_trips_and_exposes_features() -> None:
    task = generate_nl_tasks(task_types=("arithmetic",), operations=("add",), values=(2,), seed=0)[0]
    critic = DifferentialCritic()
    report = critic.compare_programs(
        VM(),
        (Instruction("CONST", dst=0, imm=1),),
        task.gold_program,
        task_prompt=task.prompt,
    )

    repair_state = build_repair_state(report)
    restored = RepairState.from_dict(repair_state.to_dict())

    assert restored == repair_state
    assert "failure=" in repair_state.to_text()
    assert len(repair_state.feature_vector()) > 0


def test_repair_vocabulary_keeps_reserved_tokens_reserved() -> None:
    task = generate_nl_tasks(task_types=("arithmetic",), operations=("add",), values=(2,), seed=0)[0]
    report = DifferentialCritic().compare_programs(
        VM(),
        (Instruction("CONST", dst=0, imm=1),),
        task.gold_program,
        task_prompt=task.prompt,
    )
    example = RepairTrainingExample(
        prompt="<pad> add <unk> add",
        repair_state=build_repair_state(report),
        target_canonical_prompt=task.canonical_prompt,
        gold_program=task.gold_program,
        task_type=task.task_type,
        corruption_name="reserved-token-prompt",
    )

    vocabulary = RepairVocabulary.from_examples((example,))

    assert vocabulary.itos[:2] == ["<pad>", "<unk>"]
    assert vocabulary.itos.count("<pad>") == 1
    assert vocabulary.itos.count("<unk>") == 1
    assert vocabulary.encode("<pad> <unk> add") == [0, 1, vocabulary.stoi["add"]]


def test_model_driven_repair_compiler_rejects_empty_training_examples() -> None:
    with pytest.raises(ValueError, match="at least one training example"):
        build_model_driven_repair_compiler(cast(BackboneConditionedCompiler, object()), ())


def test_model_driven_repair_improves_held_out_failures() -> None:
    base_compiler = _build_trained_nl_compiler()
    train_tasks = generate_nl_tasks(
        task_types=("arithmetic", "max", "sum_to_n"),
        operations=("add", "sub"),
        values=(1, 2),
        seed=1,
    )
    examples = build_repair_training_examples(train_tasks, corruption_names=("drop_halt", "swap_arithmetic", "redirect_jump"))
    repair_compiler = build_model_driven_repair_compiler(base_compiler, examples)
    metrics = repair_compiler.fit(examples, epochs=256)
    benchmark_cases = build_held_out_repair_benchmark(train_tasks, corruption_names=("shift_const",))
    controller = RepairLoopController(critic=DifferentialCritic(), max_rounds=3)

    report = run_repair_benchmark(controller, repair_compiler, benchmark_cases)
    fit_metrics = evaluate_model_driven_repair(repair_compiler, examples)

    assert metrics["canonical_accuracy"] == pytest.approx(1.0)
    assert fit_metrics.canonical_accuracy == pytest.approx(1.0)
    assert report.baseline_success_rate == pytest.approx(0.0)
    assert report.repaired_success_rate > report.baseline_success_rate
    assert report.metrics.success_after_2_rounds >= report.repaired_success_rate
    assert set(report.task_type_improvement) == {"arithmetic", "max", "sum_to_n"}
