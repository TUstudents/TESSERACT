from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable, Sequence

from tesseract.compiler.synthetic import (
    RESULT_REGISTER,
    SUPPORTED_OPERATIONS,
    SUPPORTED_TASK_TYPES,
    SyntheticTask,
    build_abs_program,
    build_abs_prompt,
    build_arithmetic_prompt,
    build_factorial_program,
    build_factorial_prompt,
    build_fibonacci_program,
    build_fibonacci_prompt,
    build_gold_program,
    build_max_program,
    build_max_prompt,
    build_memory_sum_program,
    build_memory_sum_prompt,
    build_sum_to_n_program,
    build_sum_to_n_prompt,
    evaluate_factorial,
    evaluate_fibonacci,
    evaluate_operation,
)
from tesseract.vm import Instruction


@dataclass(frozen=True)
class NaturalLanguageTask:
    prompt: str
    canonical_prompt: str
    expected_output: int
    gold_program: tuple[Instruction, ...]
    result_register: int = RESULT_REGISTER
    task_type: str = "arithmetic"
    values: tuple[int, ...] = ()

    def to_synthetic_task(self) -> SyntheticTask:
        return SyntheticTask(
            prompt=self.canonical_prompt,
            expected_output=self.expected_output,
            gold_program=self.gold_program,
            result_register=self.result_register,
            task_type=self.task_type,
            values=self.values,
            lhs=self.values[0] if len(self.values) >= 1 else None,
            rhs=self.values[1] if len(self.values) >= 2 else None,
            n=self.values[0] if len(self.values) == 1 else None,
            operation=self._operation(),
        )

    def _operation(self) -> str | None:
        if self.task_type != "arithmetic":
            return None
        parts = self.canonical_prompt.split()
        return parts[1] if len(parts) >= 2 else None


def generate_nl_tasks(
    *,
    task_types: Sequence[str] = ("arithmetic", "max", "sum_to_n"),
    operations: Sequence[str] = ("add", "sub", "mul", "div"),
    values: Iterable[int] = range(0, 6),
    result_register: int = RESULT_REGISTER,
    seed: int | None = None,
    include_all_prompt_variants: bool = False,
    memory_sequences: Sequence[Sequence[int]] | None = None,
) -> list[NaturalLanguageTask]:
    rng = random.Random(seed)
    tasks: list[NaturalLanguageTask] = []
    cached_values = list(values)
    requested_types = set(task_types)
    unknown_task_types = requested_types - SUPPORTED_TASK_TYPES
    if unknown_task_types:
        raise ValueError(f"unsupported task type(s): {sorted(unknown_task_types)!r}")
    unknown_operations = set(operations) - SUPPORTED_OPERATIONS
    if unknown_operations:
        raise ValueError(f"unsupported operation(s): {sorted(unknown_operations)!r}")

    if "arithmetic" in requested_types:
        for operation in operations:
            for lhs in cached_values:
                for rhs in cached_values:
                    if operation == "div" and rhs == 0:
                        continue
                    canonical_prompt = build_arithmetic_prompt(operation, lhs, rhs)
                    prompts = _arithmetic_templates(operation, lhs, rhs)
                    selected_prompts = prompts if include_all_prompt_variants else (rng.choice(prompts),)
                    for prompt in selected_prompts:
                        tasks.append(
                            NaturalLanguageTask(
                                prompt=prompt,
                                canonical_prompt=canonical_prompt,
                                expected_output=evaluate_operation(operation, lhs, rhs),
                                gold_program=build_gold_program(
                                    operation,
                                    lhs,
                                    rhs,
                                    result_register=result_register,
                                ),
                                result_register=result_register,
                                task_type="arithmetic",
                                values=(lhs, rhs),
                            )
                        )

    if "max" in requested_types:
        for lhs in cached_values:
            for rhs in cached_values:
                prompts = _max_templates(lhs, rhs)
                selected_prompts = prompts if include_all_prompt_variants else (rng.choice(prompts),)
                for prompt in selected_prompts:
                    tasks.append(
                        NaturalLanguageTask(
                            prompt=prompt,
                            canonical_prompt=build_max_prompt(lhs, rhs),
                            expected_output=max(lhs, rhs),
                            gold_program=build_max_program(lhs, rhs, result_register=result_register),
                            result_register=result_register,
                            task_type="max",
                            values=(lhs, rhs),
                        )
                    )

    if "sum_to_n" in requested_types:
        for n in cached_values:
            if n < 0:
                raise ValueError("sum_to_n requires a non-negative integer")
            prompts = _sum_to_n_templates(n)
            selected_prompts = prompts if include_all_prompt_variants else (rng.choice(prompts),)
            for prompt in selected_prompts:
                tasks.append(
                    NaturalLanguageTask(
                        prompt=prompt,
                        canonical_prompt=build_sum_to_n_prompt(n),
                        expected_output=n * (n + 1) // 2,
                        gold_program=build_sum_to_n_program(n, result_register=result_register),
                        result_register=result_register,
                        task_type="sum_to_n",
                        values=(n,),
                    )
                )

    if "factorial" in requested_types:
        for n in cached_values:
            if n < 0:
                raise ValueError("factorial requires a non-negative integer")
            prompts = _factorial_templates(n)
            selected_prompts = prompts if include_all_prompt_variants else (rng.choice(prompts),)
            for prompt in selected_prompts:
                tasks.append(
                    NaturalLanguageTask(
                        prompt=prompt,
                        canonical_prompt=build_factorial_prompt(n),
                        expected_output=evaluate_factorial(n),
                        gold_program=build_factorial_program(n, result_register=result_register),
                        result_register=result_register,
                        task_type="factorial",
                        values=(n,),
                    )
                )

    if "fibonacci" in requested_types:
        for n in cached_values:
            if n < 0:
                raise ValueError("fibonacci requires a non-negative integer")
            prompts = _fibonacci_templates(n)
            selected_prompts = prompts if include_all_prompt_variants else (rng.choice(prompts),)
            for prompt in selected_prompts:
                tasks.append(
                    NaturalLanguageTask(
                        prompt=prompt,
                        canonical_prompt=build_fibonacci_prompt(n),
                        expected_output=evaluate_fibonacci(n),
                        gold_program=build_fibonacci_program(n, result_register=result_register),
                        result_register=result_register,
                        task_type="fibonacci",
                        values=(n,),
                    )
                )

    if "abs" in requested_types:
        signed_values = sorted({*cached_values, *(-value for value in cached_values)})
        for value in signed_values:
            prompts = _abs_templates(value)
            selected_prompts = prompts if include_all_prompt_variants else (rng.choice(prompts),)
            for prompt in selected_prompts:
                tasks.append(
                    NaturalLanguageTask(
                        prompt=prompt,
                        canonical_prompt=build_abs_prompt(value),
                        expected_output=abs(value),
                        gold_program=build_abs_program(value, result_register=result_register),
                        result_register=result_register,
                        task_type="abs",
                        values=(value,),
                    )
                )

    if "memory_sum" in requested_types:
        sequences = memory_sequences if memory_sequences is not None else _default_memory_sequences(cached_values)
        for sequence in sequences:
            tupled = tuple(sequence)
            prompts = _memory_sum_templates(tupled)
            selected_prompts = prompts if include_all_prompt_variants else (rng.choice(prompts),)
            for prompt in selected_prompts:
                tasks.append(
                    NaturalLanguageTask(
                        prompt=prompt,
                        canonical_prompt=build_memory_sum_prompt(tupled),
                        expected_output=sum(tupled),
                        gold_program=build_memory_sum_program(tupled, result_register=result_register),
                        result_register=result_register,
                        task_type="memory_sum",
                        values=tupled,
                    )
                )

    return tasks


def _default_memory_sequences(values: Sequence[int]) -> tuple[tuple[int, ...], ...]:
    if not values:
        return ((),)
    sequences: list[tuple[int, ...]] = []
    max_length = min(3, len(values))
    for length in range(1, max_length + 1):
        sequences.append(tuple(values[:length]))
    if len(values) >= 2:
        sequences.append((values[-1], values[0]))
    deduplicated: list[tuple[int, ...]] = []
    for sequence in sequences:
        if sequence not in deduplicated:
            deduplicated.append(sequence)
    return tuple(deduplicated)


def _arithmetic_templates(operation: str, lhs: int, rhs: int) -> tuple[str, ...]:
    templates: dict[str, tuple[str, ...]] = {
        "add": (
            f"What is {lhs} plus {rhs}?",
            f"Add {lhs} and {rhs}",
        ),
        "sub": (
            f"What is {lhs} minus {rhs}?",
            f"Subtract {rhs} from {lhs}",
        ),
        "mul": (
            f"What is {lhs} times {rhs}?",
            f"Multiply {lhs} and {rhs}",
        ),
        "div": (
            f"What is {lhs} divided by {rhs}?",
            f"Divide {lhs} by {rhs}",
        ),
    }
    if operation not in templates:
        raise ValueError(f"unsupported operation {operation!r}")
    return templates[operation]


def _max_templates(lhs: int, rhs: int) -> tuple[str, ...]:
    return (
        f"Max of {lhs} and {rhs}",
        f"Which number is larger: {lhs} or {rhs}?",
        f"Compare {lhs} and {rhs} and return the larger",
    )


def _sum_to_n_templates(n: int) -> tuple[str, ...]:
    return (
        f"Sum integers from 1 to {n}",
        f"Sum all integers up to {n}",
        f"Compute the triangular number of {n}",
    )


def _factorial_templates(n: int) -> tuple[str, ...]:
    return (
        f"Factorial of {n}",
        f"Compute factorial of {n}",
        f"What is the factorial of {n}?",
    )


def _fibonacci_templates(n: int) -> tuple[str, ...]:
    return (
        f"Fibonacci of {n}",
        f"Compute fibonacci of {n}",
        f"What is fibonacci of {n}?",
    )


def _abs_templates(value: int) -> tuple[str, ...]:
    return (
        f"Absolute value of {value}",
        f"Compute absolute value of {value}",
        f"Return the absolute value of {value}",
    )


def _memory_sum_templates(values: Sequence[int]) -> tuple[str, ...]:
    joined = " ".join(str(value) for value in values)
    return (
        f"Sum memory values {joined}".strip(),
        f"Compute memory sum {joined}".strip(),
        f"Add the memory cells {joined}".strip(),
    )
