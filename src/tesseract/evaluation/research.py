from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

from tesseract.critic import (
    CriticTrainingExample,
    LearnedCritic,
    RepairBenchmarkCase,
    RepairBenchmarkReport,
    RepairLoopController,
    build_repair_training_examples,
    evaluate_learned_critic,
    generate_corrupted_programs,
    run_repair_benchmark,
)
from tesseract.compiler.nl import BackboneConditionedCompiler, RepairCapableCompiler
from tesseract.vm import Trap, VM, validate_program

import json

from .benchmark import BenchmarkReport, BenchmarkSuite, run_nl_benchmark


@dataclass(frozen=True)
class CriticLocalizationResult:
    task_prompt: str | None
    failure_type_correct: bool
    first_step_correct: bool


@dataclass(frozen=True)
class CriticLocalizationReport:
    results: tuple[CriticLocalizationResult, ...]

    @property
    def failure_type_accuracy(self) -> float:
        if not self.results:
            return 0.0
        return sum(1 for result in self.results if result.failure_type_correct) / len(self.results)

    @property
    def first_step_accuracy(self) -> float:
        if not self.results:
            return 0.0
        return sum(1 for result in self.results if result.first_step_correct) / len(self.results)

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["failure_type_accuracy"] = self.failure_type_accuracy
        payload["first_step_accuracy"] = self.first_step_accuracy
        return payload


@dataclass(frozen=True)
class AntiShortcutResult:
    prompt: str
    task_type: str
    exact_output: bool
    corrupted_output_matches_expected: bool
    corrupted_trap_kind: str | None


@dataclass(frozen=True)
class AntiShortcutReport:
    suite_name: str
    seed: int
    results: tuple[AntiShortcutResult, ...]

    @property
    def faithful_execution_rate(self) -> float:
        if not self.results:
            return 0.0
        return sum(1 for result in self.results if result.exact_output) / len(self.results)

    @property
    def corrupted_program_accuracy(self) -> float:
        if not self.results:
            return 0.0
        return sum(1 for result in self.results if result.corrupted_output_matches_expected) / len(self.results)

    @property
    def degradation(self) -> float:
        return self.faithful_execution_rate - self.corrupted_program_accuracy

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["faithful_execution_rate"] = self.faithful_execution_rate
        payload["corrupted_program_accuracy"] = self.corrupted_program_accuracy
        payload["degradation"] = self.degradation
        return payload


@dataclass(frozen=True)
class ExperimentManifest:
    experiment_name: str
    seed: int
    suite_payload: str
    model_config: dict[str, Any]
    checkpoint_metadata: dict[str, Any] = field(default_factory=dict)
    code_identifiers: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "experiment_name": self.experiment_name,
            "seed": self.seed,
            "suite_payload": self.suite_payload,
            "model_config": dict(self.model_config),
            "checkpoint_metadata": dict(self.checkpoint_metadata),
            "code_identifiers": dict(self.code_identifiers),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExperimentManifest:
        return cls(
            experiment_name=data["experiment_name"],
            seed=int(data["seed"]),
            suite_payload=str(data["suite_payload"]),
            model_config=dict(data.get("model_config", {})),
            checkpoint_metadata=dict(data.get("checkpoint_metadata", {})),
            code_identifiers=dict(data.get("code_identifiers", {})),
        )


@dataclass(frozen=True)
class ResearchEvaluationReport:
    manifest: ExperimentManifest
    exact_execution: BenchmarkReport
    critic_localization: CriticLocalizationReport | None = None
    repair_improvement: RepairBenchmarkReport | None = None
    anti_shortcut: AntiShortcutReport | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "manifest": self.manifest.to_dict(),
            "exact_execution": self.exact_execution.to_dict(),
            "critic_localization": None if self.critic_localization is None else self.critic_localization.to_dict(),
            "repair_improvement": None if self.repair_improvement is None else self.repair_improvement.to_dict(),
            "anti_shortcut": None if self.anti_shortcut is None else self.anti_shortcut.to_dict(),
        }


def build_experiment_manifest(
    *,
    experiment_name: str,
    seed: int,
    suite: BenchmarkSuite,
    model_config: dict[str, Any],
    checkpoint_metadata: dict[str, Any] | None = None,
    code_identifiers: dict[str, str] | None = None,
) -> ExperimentManifest:
    return ExperimentManifest(
        experiment_name=experiment_name,
        seed=seed,
        suite_payload=json.dumps(suite.to_dict(), sort_keys=True),
        model_config=model_config,
        checkpoint_metadata={} if checkpoint_metadata is None else checkpoint_metadata,
        code_identifiers={} if code_identifiers is None else code_identifiers,
    )


def run_critic_localization_benchmark(
    critic: LearnedCritic,
    examples: Sequence[CriticTrainingExample],
) -> CriticLocalizationReport:
    metrics = evaluate_learned_critic(critic, examples)
    if not examples:
        return CriticLocalizationReport(results=())
    results = tuple(
        CriticLocalizationResult(
            task_prompt=example.task_prompt,
            failure_type_correct=critic._predict(example.features).failure_type == example.failure_type,
            first_step_correct=critic._predict(example.features).first_failing_step == example.first_failing_step,
        )
        for example in examples
    )
    assert metrics.failure_type_accuracy >= 0.0
    assert metrics.first_step_accuracy >= 0.0
    return CriticLocalizationReport(results=results)


def run_anti_shortcut_benchmark(
    compiler: BackboneConditionedCompiler,
    suite: BenchmarkSuite,
    *,
    vm: VM | None = None,
    corruption_names: Sequence[str] = ("shift_const", "swap_arithmetic", "redirect_jump"),
) -> AntiShortcutReport:
    machine = vm if vm is not None else VM()
    exact_report = run_nl_benchmark(compiler, suite, vm=machine)
    results: list[AntiShortcutResult] = []
    for task, result in zip(suite.tasks, exact_report.results, strict=True):
        compile_result = compiler.compile_with_backbone_output(task.prompt)
        corruptions = generate_corrupted_programs(compile_result.program, corruption_names=corruption_names)
        corrupted_matches = False
        corrupted_trap: str | None = None
        if corruptions:
            candidate = corruptions[0].program
            try:
                validate_program(candidate)
                try:
                    state = machine.execute(candidate)
                    corrupted_matches = state.registers.get(task.result_register) == task.expected_output
                except Trap as trap:
                    corrupted_trap = trap.kind
            except Exception:
                corrupted_trap = "VALIDATION_ERROR"
        results.append(
            AntiShortcutResult(
                prompt=task.prompt,
                task_type=task.task_type,
                exact_output=result.observed_output == task.expected_output,
                corrupted_output_matches_expected=corrupted_matches,
                corrupted_trap_kind=corrupted_trap,
            )
        )
    return AntiShortcutReport(suite_name=suite.name, seed=suite.seed, results=tuple(results))


def run_research_evaluation(
    *,
    manifest: ExperimentManifest,
    compiler: BackboneConditionedCompiler,
    suite: BenchmarkSuite,
    critic_examples: Sequence[CriticTrainingExample] | None = None,
    critic: LearnedCritic | None = None,
    repair_controller: RepairLoopController | None = None,
    repair_compiler: RepairCapableCompiler | None = None,
    repair_cases: Sequence[RepairBenchmarkCase] | None = None,
    vm: VM | None = None,
) -> ResearchEvaluationReport:
    exact_execution = run_nl_benchmark(compiler, suite, vm=vm)
    critic_localization = None
    if critic is not None and critic_examples is not None:
        critic_localization = run_critic_localization_benchmark(critic, critic_examples)
    repair_improvement = None
    if repair_controller is not None and repair_compiler is not None and repair_cases is not None:
        repair_improvement = run_repair_benchmark(repair_controller, repair_compiler, repair_cases)
    anti_shortcut = run_anti_shortcut_benchmark(compiler, suite, vm=vm)
    return ResearchEvaluationReport(
        manifest=manifest,
        exact_execution=exact_execution,
        critic_localization=critic_localization,
        repair_improvement=repair_improvement,
        anti_shortcut=anti_shortcut,
    )


def build_repair_improvement_benchmark(
    controller: RepairLoopController,
    compiler: RepairCapableCompiler,
    cases: Sequence[RepairBenchmarkCase],
) -> RepairBenchmarkReport:
    return run_repair_benchmark(controller, compiler, cases)


def build_repair_examples_from_suite(suite: BenchmarkSuite) -> list[Any]:
    return build_repair_training_examples(suite.tasks)
