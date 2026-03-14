from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from tesseract.backbone.datasets import NaturalLanguageTask, generate_nl_tasks
from tesseract.compiler.nl import BackboneConditionedCompiler
from tesseract.compiler.synthetic import RESULT_REGISTER
from tesseract.vm import Trap, VM, ValidationError, program_from_dict, program_to_dict, validate_program


@dataclass(frozen=True)
class BenchmarkSuite:
    name: str
    seed: int
    tasks: tuple[NaturalLanguageTask, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "seed": self.seed,
            "tasks": [
                {
                    "prompt": task.prompt,
                    "canonical_prompt": task.canonical_prompt,
                    "expected_output": task.expected_output,
                    "gold_program": program_to_dict(task.gold_program),
                    "result_register": task.result_register,
                    "task_type": task.task_type,
                    "values": list(task.values),
                }
                for task in self.tasks
            ],
        }


@dataclass(frozen=True)
class BenchmarkResult:
    prompt: str
    canonical_prompt: str
    task_type: str
    expected_output: int
    observed_output: int | None
    valid_program: bool
    execution_success: bool
    exact_program_match: bool
    program_length: int
    trap_kind: str | None = None


@dataclass(frozen=True)
class BenchmarkReport:
    suite_name: str
    seed: int
    results: tuple[BenchmarkResult, ...] = field(default_factory=tuple)

    @property
    def exact_output_accuracy(self) -> float:
        if not self.results:
            return 0.0
        matches = sum(1 for result in self.results if result.observed_output == result.expected_output)
        return matches / len(self.results)

    @property
    def compile_validity_rate(self) -> float:
        if not self.results:
            return 0.0
        return sum(1 for result in self.results if result.valid_program) / len(self.results)

    @property
    def execution_success_rate(self) -> float:
        if not self.results:
            return 0.0
        return sum(1 for result in self.results if result.execution_success) / len(self.results)

    @property
    def exact_program_match(self) -> float:
        if not self.results:
            return 0.0
        return sum(1 for result in self.results if result.exact_program_match) / len(self.results)

    @property
    def average_program_length(self) -> float:
        if not self.results:
            return 0.0
        return sum(result.program_length for result in self.results) / len(self.results)

    def task_type_metrics(self) -> dict[str, dict[str, float]]:
        metrics: dict[str, dict[str, float]] = {}
        task_types = sorted({result.task_type for result in self.results})
        for task_type in task_types:
            matching = [result for result in self.results if result.task_type == task_type]
            total = len(matching)
            metrics[task_type] = {
                "exact_output_accuracy": sum(
                    1 for result in matching if result.observed_output == result.expected_output
                )
                / total,
                "execution_success_rate": sum(1 for result in matching if result.execution_success) / total,
                "exact_program_match": sum(1 for result in matching if result.exact_program_match) / total,
            }
        return metrics

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["exact_output_accuracy"] = self.exact_output_accuracy
        payload["compile_validity_rate"] = self.compile_validity_rate
        payload["execution_success_rate"] = self.execution_success_rate
        payload["exact_program_match"] = self.exact_program_match
        payload["average_program_length"] = self.average_program_length
        payload["task_type_metrics"] = self.task_type_metrics()
        return payload


def benchmark_suite_from_dict(data: dict[str, Any]) -> BenchmarkSuite:
    tasks = tuple(
        NaturalLanguageTask(
            prompt=task_data["prompt"],
            canonical_prompt=task_data["canonical_prompt"],
            expected_output=task_data["expected_output"],
            gold_program=tuple(program_from_dict(task_data["gold_program"])),
            result_register=task_data.get("result_register", RESULT_REGISTER),
            task_type=task_data.get("task_type", "arithmetic"),
            values=tuple(task_data.get("values", [])),
        )
        for task_data in data.get("tasks", [])
    )
    return BenchmarkSuite(
        name=data["name"],
        seed=data["seed"],
        tasks=tasks,
    )


def build_nl_benchmark_suite(
    *,
    name: str = "nl_core",
    seed: int = 0,
) -> BenchmarkSuite:
    tasks = tuple(
        generate_nl_tasks(
            task_types=("arithmetic", "max", "sum_to_n", "factorial", "fibonacci", "abs", "memory_sum"),
            operations=("add", "sub"),
            values=(1, 2),
            seed=seed,
        )
    )
    return BenchmarkSuite(name=name, seed=seed, tasks=tasks)


def run_nl_benchmark(
    compiler: BackboneConditionedCompiler,
    suite: BenchmarkSuite,
    *,
    vm: VM | None = None,
) -> BenchmarkReport:
    machine = vm if vm is not None else VM()
    results: list[BenchmarkResult] = []
    for task in suite.tasks:
        compile_result = compiler.compile_with_backbone_output(task.prompt)
        program = tuple(compile_result.program)
        valid_program = True
        observed_output: int | None = None
        execution_success = False
        trap_kind: str | None = None
        try:
            validate_program(program)
        except ValidationError:
            valid_program = False
        if valid_program:
            try:
                state = machine.execute(program)
                observed_output = state.registers.get(task.result_register, 0)
                execution_success = True
            except Trap as trap:
                trap_kind = trap.kind
        results.append(
            BenchmarkResult(
                prompt=task.prompt,
                canonical_prompt=compile_result.backbone_output.canonical_prompt,
                task_type=task.task_type,
                expected_output=task.expected_output,
                observed_output=observed_output,
                valid_program=valid_program,
                execution_success=execution_success,
                exact_program_match=program == task.gold_program,
                program_length=len(program),
                trap_kind=trap_kind,
            )
        )
    return BenchmarkReport(suite_name=suite.name, seed=suite.seed, results=tuple(results))
