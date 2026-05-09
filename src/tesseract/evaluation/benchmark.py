from __future__ import annotations

from dataclasses import asdict, dataclass, field
from time import perf_counter
from typing import Any

from tesseract.backbone.datasets import NaturalLanguageTask, generate_nl_tasks
from tesseract.compiler.nl import BackboneConditionedCompiler
from tesseract.compiler.synthetic import RESULT_REGISTER
from tesseract.vm import Trap, VM, VMState, ValidationError, program_from_dict, program_to_dict, validate_program


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
    observed_output: int | bool | float | None
    valid_program: bool
    execution_success: bool
    exact_program_match: bool
    program_length: int
    trap_kind: str | None = None
    compile_failure_kind: str | None = None
    trace_length: int = 0
    gold_trace_length: int = 0
    macro_step_efficiency: float = 0.0
    compile_time_ms: float = 0.0
    execute_time_ms: float = 0.0


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

    @property
    def shortcut_rate(self) -> float:
        if not self.results:
            return 0.0
        shortcuts = sum(
            1
            for result in self.results
            if result.observed_output == result.expected_output and not result.exact_program_match
        )
        return shortcuts / len(self.results)

    @property
    def macro_step_efficiency(self) -> float:
        if not self.results:
            return 0.0
        return sum(result.macro_step_efficiency for result in self.results) / len(self.results)

    def compile_failure_breakdown(self) -> dict[str, float]:
        return self._breakdown(result.compile_failure_kind for result in self.results if result.compile_failure_kind is not None)

    def execution_failure_breakdown(self) -> dict[str, float]:
        return self._breakdown(result.trap_kind for result in self.results if result.trap_kind is not None)

    def trace_length_summary(self) -> dict[str, float]:
        if not self.results:
            return {"average_trace_length": 0.0, "average_gold_trace_length": 0.0, "max_trace_length": 0.0}
        return {
            "average_trace_length": sum(result.trace_length for result in self.results) / len(self.results),
            "average_gold_trace_length": sum(result.gold_trace_length for result in self.results) / len(self.results),
            "max_trace_length": float(max(result.trace_length for result in self.results)),
        }

    def performance_summary(self) -> dict[str, float]:
        if not self.results:
            return {"average_compile_time_ms": 0.0, "average_execute_time_ms": 0.0}
        return {
            "average_compile_time_ms": sum(result.compile_time_ms for result in self.results) / len(self.results),
            "average_execute_time_ms": sum(result.execute_time_ms for result in self.results) / len(self.results),
        }

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
                "macro_step_efficiency": sum(result.macro_step_efficiency for result in matching) / total,
            }
        return metrics

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["exact_output_accuracy"] = self.exact_output_accuracy
        payload["compile_validity_rate"] = self.compile_validity_rate
        payload["execution_success_rate"] = self.execution_success_rate
        payload["exact_program_match"] = self.exact_program_match
        payload["average_program_length"] = self.average_program_length
        payload["shortcut_rate"] = self.shortcut_rate
        payload["macro_step_efficiency"] = self.macro_step_efficiency
        payload["compile_failure_breakdown"] = self.compile_failure_breakdown()
        payload["execution_failure_breakdown"] = self.execution_failure_breakdown()
        payload["trace_length_summary"] = self.trace_length_summary()
        payload["performance_summary"] = self.performance_summary()
        payload["task_type_metrics"] = self.task_type_metrics()
        return payload

    def _breakdown(self, labels: Any) -> dict[str, float]:
        counts: dict[str, int] = {}
        total = 0
        for label in labels:
            counts[str(label)] = counts.get(str(label), 0) + 1
            total += 1
        if total == 0:
            return {}
        return {label: count / total for label, count in sorted(counts.items())}


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


def build_anti_shortcut_benchmark_suite(
    *,
    name: str = "anti_shortcut",
    seed: int = 0,
) -> BenchmarkSuite:
    tasks = tuple(
        generate_nl_tasks(
            task_types=("arithmetic", "max", "sum_to_n", "factorial", "fibonacci", "abs", "memory_sum"),
            operations=("add", "sub"),
            values=(1, 2),
            include_all_prompt_variants=True,
            seed=seed,
        )
    )
    return BenchmarkSuite(name=name, seed=seed, tasks=tasks)


def build_macro_step_benchmark_suite(
    *,
    name: str = "macro_step",
    seed: int = 0,
) -> BenchmarkSuite:
    tasks = tuple(
        generate_nl_tasks(
            task_types=("sum_to_n", "factorial", "fibonacci", "memory_sum"),
            operations=("add",),
            values=(1, 2, 3),
            include_all_prompt_variants=True,
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
    gold_trace_lengths = tuple(_trace_length(machine, task.gold_program) for task in suite.tasks)
    results: list[BenchmarkResult] = []
    for index, task in enumerate(suite.tasks):
        compile_started = perf_counter()
        try:
            compile_result = compiler.compile_with_backbone_output(task.prompt)
        except Exception as error:
            compile_time_ms = (perf_counter() - compile_started) * 1000.0
            results.append(
                BenchmarkResult(
                    prompt=task.prompt,
                    canonical_prompt=task.canonical_prompt,
                    task_type=task.task_type,
                    expected_output=task.expected_output,
                    observed_output=None,
                    valid_program=False,
                    execution_success=False,
                    exact_program_match=False,
                    program_length=0,
                    compile_failure_kind=f"COMPILE_ERROR:{type(error).__name__}",
                    gold_trace_length=gold_trace_lengths[index],
                    compile_time_ms=compile_time_ms,
                )
            )
            continue
        compile_time_ms = (perf_counter() - compile_started) * 1000.0
        program = tuple(compile_result.program)
        valid_program = True
        compile_failure_kind: str | None = None
        observed_output: int | bool | float | None = None
        execution_success = False
        trap_kind: str | None = None
        trace_length = 0
        execute_time_ms = 0.0
        try:
            validate_program(program, register_count=machine.register_count)
        except ValidationError:
            valid_program = False
            compile_failure_kind = "VALIDATION_ERROR"
        if valid_program:
            execute_started = perf_counter()
            final_state, trap_kind = _execute_with_trace(machine, program)
            execute_time_ms = (perf_counter() - execute_started) * 1000.0
            trace_length = len(final_state.trace)
            if trap_kind is None:
                observed_output = final_state.registers.get(task.result_register, 0)
                execution_success = True
        gold_trace_length = gold_trace_lengths[index]
        macro_step_efficiency = (gold_trace_length / trace_length) if trace_length else 0.0
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
                compile_failure_kind=compile_failure_kind,
                trace_length=trace_length,
                gold_trace_length=gold_trace_length,
                macro_step_efficiency=macro_step_efficiency,
                compile_time_ms=compile_time_ms,
                execute_time_ms=execute_time_ms,
            )
        )
    return BenchmarkReport(suite_name=suite.name, seed=suite.seed, results=tuple(results))


def _execute_with_trace(machine: VM, program: tuple[Any, ...]) -> tuple[VMState, str | None]:
    state = VMState()
    try:
        final_state = machine.execute(program, state=state, trace=True)
        return final_state, None
    except Trap as trap:
        return state, trap.kind


def _trace_length(machine: VM, program: tuple[Any, ...]) -> int:
    state, _ = _execute_with_trace(machine, program)
    return len(state.trace)
