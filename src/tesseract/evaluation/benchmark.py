from __future__ import annotations

from dataclasses import asdict, dataclass, field

from tesseract.backbone.datasets import NaturalLanguageTask, generate_nl_tasks
from tesseract.compiler.nl import BackboneConditionedCompiler
from tesseract.vm import Trap, VM, ValidationError, validate_program


@dataclass(frozen=True)
class BenchmarkSuite:
    name: str
    seed: int
    tasks: tuple[NaturalLanguageTask, ...]


@dataclass(frozen=True)
class BenchmarkResult:
    prompt: str
    canonical_prompt: str
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

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["exact_output_accuracy"] = self.exact_output_accuracy
        payload["compile_validity_rate"] = self.compile_validity_rate
        payload["execution_success_rate"] = self.execution_success_rate
        payload["exact_program_match"] = self.exact_program_match
        payload["average_program_length"] = self.average_program_length
        return payload


def build_nl_benchmark_suite(
    *,
    name: str = "nl_core",
    seed: int = 0,
) -> BenchmarkSuite:
    tasks = tuple(
        generate_nl_tasks(
            task_types=("arithmetic", "max", "sum_to_n"),
            operations=("add", "sub"),
            values=(1, 2, 3),
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
