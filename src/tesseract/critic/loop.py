from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Sequence

from tesseract.backbone.datasets import NaturalLanguageTask
from tesseract.compiler.nl import NaturalLanguageCompileResult, RepairCapableCompiler
from tesseract.vm import Instruction, VM, ValidationError, validate_program

from .differential import DifferentialCritic
from .schema import CriticReport

RepairTermination = Literal["success", "max_rounds", "oscillation"]


@dataclass(frozen=True)
class RepairContext:
    task_prompt: str
    candidate_program: tuple[Instruction, ...]
    critic_report: CriticReport
    round_index: int


@dataclass(frozen=True)
class RepairAttempt:
    round_index: int
    compile_result: NaturalLanguageCompileResult
    critic_report: CriticReport
    repair_context: RepairContext | None = None


@dataclass(frozen=True)
class RepairLoopResult:
    success: bool
    termination_reason: RepairTermination
    attempts: tuple[RepairAttempt, ...] = ()

    @property
    def rounds_used(self) -> int:
        return len(self.attempts)

    @property
    def final_program(self) -> tuple[Instruction, ...]:
        if not self.attempts:
            return ()
        return self.attempts[-1].compile_result.program

    @property
    def final_report(self) -> CriticReport | None:
        if not self.attempts:
            return None
        return self.attempts[-1].critic_report


@dataclass(frozen=True)
class RepairLoopMetrics:
    success_after_1_round: float
    success_after_2_rounds: float
    success_after_3_rounds: float
    non_convergence_rate: float
    oscillation_rate: float
    average_rounds: float
    average_extra_steps: float


def build_repair_context(
    *,
    task_prompt: str,
    candidate_program: tuple[Instruction, ...],
    critic_report: CriticReport,
    round_index: int,
) -> RepairContext:
    return RepairContext(
        task_prompt=task_prompt,
        candidate_program=candidate_program,
        critic_report=critic_report,
        round_index=round_index,
    )


@dataclass
class RepairLoopController:
    critic: DifferentialCritic
    max_rounds: int = 3
    vm: VM = field(default_factory=VM)

    def run(
        self,
        task: NaturalLanguageTask,
        compiler: RepairCapableCompiler,
    ) -> RepairLoopResult:
        attempts: list[RepairAttempt] = []
        seen_programs: set[tuple[Instruction, ...]] = set()

        for round_index in range(self.max_rounds):
            if round_index == 0:
                compile_result = compiler.compile_with_backbone_output(task.prompt)
            else:
                previous_context = attempts[-1].repair_context
                previous_report = attempts[-1].critic_report if previous_context is None else previous_context.critic_report
                compile_result = compiler.repair_compile(task.prompt, previous_report)

            program = tuple(compile_result.program)
            report = self.critic.compare_programs(
                self.vm,
                program,
                task.gold_program,
                task_prompt=task.prompt,
            )
            repair_context = build_repair_context(
                task_prompt=task.prompt,
                candidate_program=program,
                critic_report=report,
                round_index=round_index,
            )
            attempts.append(
                RepairAttempt(
                    round_index=round_index,
                    compile_result=compile_result,
                    critic_report=report,
                    repair_context=repair_context,
                )
            )

            if report.failure_type == "SUCCESS":
                return RepairLoopResult(
                    success=True,
                    termination_reason="success",
                    attempts=tuple(attempts),
                )

            if program in seen_programs:
                return RepairLoopResult(
                    success=False,
                    termination_reason="oscillation",
                    attempts=tuple(attempts),
                )
            seen_programs.add(program)

            try:
                validate_program(program)
            except ValidationError:
                continue

        return RepairLoopResult(
            success=False,
            termination_reason="max_rounds",
            attempts=tuple(attempts),
        )


def evaluate_repair_loop(results: Sequence[RepairLoopResult]) -> RepairLoopMetrics:
    if not results:
        return RepairLoopMetrics(
            success_after_1_round=0.0,
            success_after_2_rounds=0.0,
            success_after_3_rounds=0.0,
            non_convergence_rate=0.0,
            oscillation_rate=0.0,
            average_rounds=0.0,
            average_extra_steps=0.0,
        )

    total = len(results)
    success_after_1 = sum(1 for result in results if result.success and result.rounds_used <= 1)
    success_after_2 = sum(1 for result in results if result.success and result.rounds_used <= 2)
    success_after_3 = sum(1 for result in results if result.success and result.rounds_used <= 3)
    oscillations = sum(1 for result in results if result.termination_reason == "oscillation")
    non_convergence = sum(1 for result in results if not result.success)
    average_rounds = sum(result.rounds_used for result in results) / total
    average_extra_steps = sum(max(0, result.rounds_used - 1) for result in results) / total
    return RepairLoopMetrics(
        success_after_1_round=success_after_1 / total,
        success_after_2_rounds=success_after_2 / total,
        success_after_3_rounds=success_after_3 / total,
        non_convergence_rate=non_convergence / total,
        oscillation_rate=oscillations / total,
        average_rounds=average_rounds,
        average_extra_steps=average_extra_steps,
    )
