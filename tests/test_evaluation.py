from __future__ import annotations

import json
from dataclasses import dataclass
from types import MethodType
from typing import Any, cast

import numpy as np
import pytest
import torch

from tesseract.backbone.datasets import NaturalLanguageTask
from tesseract.backbone.interface import BackboneOutput
from tesseract.backbone import RuleBasedBackbone, generate_nl_tasks
from tesseract.compiler import build_training_batch, build_vocabularies, train_step
from tesseract.compiler.nl import BackboneConditionedCompiler, NaturalLanguageCompileResult
from tesseract.compiler.synthetic import RESULT_REGISTER, make_max_task, make_sum_to_n_task, make_synthetic_task
from tesseract.critic import (
    DifferentialCritic,
    build_critic_training_examples,
    build_held_out_repair_benchmark,
    build_learned_critic,
    build_model_driven_repair_compiler,
    build_repair_training_examples,
)
from tesseract.critic.loop import RepairLoopController
from tesseract.vm import Instruction, VM
from tesseract.evaluation import (
    BenchmarkSuite,
    benchmark_report_to_json,
    benchmark_report_to_text,
    benchmark_suite_from_json,
    benchmark_suite_to_json,
    build_anti_shortcut_benchmark_suite,
    build_experiment_manifest,
    build_macro_step_benchmark_suite,
    build_nl_benchmark_suite,
    experiment_manifest_from_json,
    experiment_manifest_to_json,
    research_evaluation_report_to_json,
    research_evaluation_report_to_text,
    run_anti_shortcut_benchmark,
    run_critic_localization_benchmark,
    run_nl_benchmark,
    run_research_evaluation,
    set_global_seed,
)


def _build_trained_nl_compiler() -> BackboneConditionedCompiler:
    nl_tasks = generate_nl_tasks(
        task_types=("arithmetic", "max", "sum_to_n", "factorial", "fibonacci", "abs", "memory_sum"),
        operations=("add", "sub"),
        values=(1, 2),
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


def _build_trained_learned_critic():
    tasks = [
        make_synthetic_task("add", 2, 3),
        make_synthetic_task("sub", 5, 1),
        make_max_task(3, 1),
        make_sum_to_n_task(3),
    ]
    examples = build_critic_training_examples(tasks)
    critic = build_learned_critic()
    critic.fit(examples, epochs=256)
    return critic, examples


@dataclass
class StaticBenchmarkCompiler:
    program: tuple[Instruction, ...]

    def compile_with_backbone_output(
        self,
        prompt: str,
        *,
        repair_context: Any | None = None,
    ) -> NaturalLanguageCompileResult:
        del repair_context
        return NaturalLanguageCompileResult(
            backbone_output=BackboneOutput(
                original_prompt=prompt,
                canonical_prompt=prompt,
                task_type="arithmetic",
                result_register=RESULT_REGISTER,
            ),
            program=self.program,
        )


class RaisingBenchmarkCompiler:
    def compile_with_backbone_output(
        self,
        prompt: str,
        *,
        repair_context: Any | None = None,
    ) -> NaturalLanguageCompileResult:
        del prompt, repair_context
        raise ValueError("compile failed")


def _benchmark_task(
    *,
    prompt: str = "same prompt",
    expected_output: int = 0,
    gold_program: tuple[Instruction, ...] = (Instruction("HALT"),),
) -> NaturalLanguageTask:
    return NaturalLanguageTask(
        prompt=prompt,
        canonical_prompt=prompt,
        expected_output=expected_output,
        gold_program=gold_program,
        result_register=RESULT_REGISTER,
        task_type="arithmetic",
        values=(),
    )


def test_set_global_seed_is_reproducible() -> None:
    set_global_seed(123)
    first_torch = torch.rand(4)
    first_numpy = np.random.rand(4)
    set_global_seed(123)
    second_torch = torch.rand(4)
    second_numpy = np.random.rand(4)

    assert torch.equal(first_torch, second_torch)
    assert np.array_equal(first_numpy, second_numpy)


def test_nl_benchmark_suite_is_seed_reproducible() -> None:
    first = build_nl_benchmark_suite(seed=7)
    second = build_nl_benchmark_suite(seed=7)
    third = build_nl_benchmark_suite(seed=8)

    assert first.tasks == second.tasks
    assert first.tasks != third.tasks


def test_benchmark_suite_round_trips_through_json_freeze_payload() -> None:
    suite = build_nl_benchmark_suite(seed=7)

    payload = benchmark_suite_to_json(suite)
    restored = benchmark_suite_from_json(payload)

    assert restored == suite


def test_run_nl_benchmark_reports_exact_execution_metrics() -> None:
    compiler = _build_trained_nl_compiler()
    suite = build_nl_benchmark_suite(seed=0)

    report = run_nl_benchmark(compiler, suite)

    assert report.exact_output_accuracy == 1.0
    assert report.compile_validity_rate == 1.0
    assert report.execution_success_rate == 1.0
    assert report.exact_program_match == 1.0
    assert report.average_program_length > 4.0
    assert set(report.task_type_metrics()) == {"arithmetic", "max", "sum_to_n", "factorial", "fibonacci", "abs", "memory_sum"}


def test_run_nl_benchmark_records_timeout_failures_stably() -> None:
    compiler = _build_trained_nl_compiler()
    suite = build_nl_benchmark_suite(seed=0)

    def looping_compile_conditioned(self: Any, prompt: str, conditioning=None):
        del prompt, conditioning
        return (Instruction("JMP", imm=0), Instruction("HALT"))

    cast(Any, compiler.compiler).compile_conditioned = MethodType(looping_compile_conditioned, compiler.compiler)
    report = run_nl_benchmark(compiler, suite, vm=VM(step_budget=5))

    assert report.compile_validity_rate == 1.0
    assert report.execution_success_rate == 0.0
    assert {result.trap_kind for result in report.results} == {"TIMEOUT"}


def test_run_nl_benchmark_records_compile_exceptions() -> None:
    suite = BenchmarkSuite(name="compile_failure", seed=0, tasks=(_benchmark_task(),))

    report = run_nl_benchmark(cast(BackboneConditionedCompiler, RaisingBenchmarkCompiler()), suite)

    result = report.results[0]
    assert result.valid_program is False
    assert result.execution_success is False
    assert result.compile_failure_kind == "COMPILE_ERROR:ValueError"
    assert result.gold_trace_length == 1
    assert report.compile_validity_rate == 0.0


def test_run_nl_benchmark_tracks_duplicate_prompt_gold_traces_by_task() -> None:
    first = _benchmark_task(prompt="same prompt", gold_program=(Instruction("HALT"),))
    second = _benchmark_task(
        prompt="same prompt",
        expected_output=1,
        gold_program=(Instruction("CONST", dst=RESULT_REGISTER, imm=1), Instruction("HALT")),
    )
    suite = BenchmarkSuite(name="duplicate_prompts", seed=0, tasks=(first, second))
    compiler = StaticBenchmarkCompiler(program=(Instruction("HALT"),))

    report = run_nl_benchmark(cast(BackboneConditionedCompiler, compiler), suite)

    assert [result.gold_trace_length for result in report.results] == [1, 2]


def test_run_nl_benchmark_validates_against_vm_register_count() -> None:
    suite = BenchmarkSuite(name="small_vm", seed=0, tasks=(_benchmark_task(),))
    compiler = StaticBenchmarkCompiler(
        program=(Instruction("CONST", dst=3, imm=1), Instruction("HALT")),
    )

    report = run_nl_benchmark(cast(BackboneConditionedCompiler, compiler), suite, vm=VM(register_count=2))

    result = report.results[0]
    assert result.valid_program is False
    assert result.compile_failure_kind == "VALIDATION_ERROR"
    assert result.execution_success is False


def test_benchmark_report_serialization_helpers() -> None:
    compiler = _build_trained_nl_compiler()
    suite = build_nl_benchmark_suite(seed=0)
    report = run_nl_benchmark(compiler, suite)

    payload = benchmark_report_to_json(report)
    text = benchmark_report_to_text(report)

    parsed = json.loads(payload)
    assert parsed["suite_name"] == suite.name
    assert "exact_output_accuracy" in parsed
    assert "task_type_metrics" in parsed
    assert "performance_summary" in parsed
    assert f"suite: {suite.name}" in text
    assert "execution_success_rate:" in text
    assert "trace_lengths:" in text
    assert "task_type[factorial]:" in text


def test_benchmark_suites_cover_anti_shortcut_and_macro_step_modes() -> None:
    anti_shortcut_suite = build_anti_shortcut_benchmark_suite(seed=4)
    macro_step_suite = build_macro_step_benchmark_suite(seed=4)

    assert len(anti_shortcut_suite.tasks) > len(build_nl_benchmark_suite(seed=4).tasks)
    assert {task.task_type for task in macro_step_suite.tasks} == {"sum_to_n", "factorial", "fibonacci", "memory_sum"}


def test_anti_shortcut_benchmark_detects_corruption_degradation() -> None:
    compiler = _build_trained_nl_compiler()
    suite = build_anti_shortcut_benchmark_suite(seed=0)

    report = run_anti_shortcut_benchmark(compiler, suite)

    assert report.faithful_execution_rate == pytest.approx(1.0)
    assert report.degradation > 0.0


def test_anti_shortcut_benchmark_records_compile_exceptions() -> None:
    suite = BenchmarkSuite(name="anti_shortcut_compile_failure", seed=0, tasks=(_benchmark_task(),))

    report = run_anti_shortcut_benchmark(cast(BackboneConditionedCompiler, RaisingBenchmarkCompiler()), suite)

    result = report.results[0]
    assert result.exact_output is False
    assert result.corrupted_output_matches_expected is False
    assert result.corrupted_trap_kind == "COMPILE_ERROR:ValueError"


def test_critic_localization_benchmark_reports_accuracy() -> None:
    critic, examples = _build_trained_learned_critic()

    report = run_critic_localization_benchmark(critic, examples)

    assert report.failure_type_accuracy == pytest.approx(1.0)
    assert report.first_step_accuracy == pytest.approx(1.0)


def test_research_evaluation_captures_manifest_and_subreports() -> None:
    compiler = _build_trained_nl_compiler()
    suite = build_nl_benchmark_suite(seed=0)
    manifest = build_experiment_manifest(
        experiment_name="nl_research",
        seed=0,
        suite=suite,
        model_config={"compiler": "autoregressive", "backbone": "rule_based"},
        checkpoint_metadata={"checkpoint": "none"},
        code_identifiers={"revision": "test"},
    )
    learned_critic, critic_examples = _build_trained_learned_critic()
    repair_examples = build_repair_training_examples(list(suite.tasks), corruption_names=("drop_halt", "swap_arithmetic", "redirect_jump"))
    repair_compiler = build_model_driven_repair_compiler(compiler, repair_examples)
    repair_compiler.fit(repair_examples, epochs=256)
    repair_cases = build_held_out_repair_benchmark(list(suite.tasks), corruption_names=("shift_const",))
    controller = RepairLoopController(critic=DifferentialCritic(), max_rounds=3)

    report = run_research_evaluation(
        manifest=manifest,
        compiler=compiler,
        suite=suite,
        critic_examples=critic_examples,
        critic=learned_critic,
        repair_controller=controller,
        repair_compiler=repair_compiler,
        repair_cases=repair_cases,
    )
    manifest_payload = experiment_manifest_to_json(manifest)
    manifest_roundtrip = experiment_manifest_from_json(manifest_payload)
    report_payload = research_evaluation_report_to_json(report)
    report_text = research_evaluation_report_to_text(report)

    assert manifest_roundtrip == manifest
    parsed = json.loads(report_payload)
    assert parsed["manifest"]["experiment_name"] == "nl_research"
    assert parsed["exact_execution"]["exact_output_accuracy"] == 1.0
    assert parsed["critic_localization"]["failure_type_accuracy"] == 1.0
    assert parsed["repair_improvement"]["repaired_success_rate"] >= parsed["repair_improvement"]["baseline_success_rate"]
    assert parsed["anti_shortcut"]["degradation"] > 0.0
    assert "[critic_localization]" in report_text
    assert "[repair_improvement]" in report_text
    assert "[anti_shortcut]" in report_text
