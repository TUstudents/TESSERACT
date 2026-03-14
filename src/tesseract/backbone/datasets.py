from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable, Sequence

from tesseract.compiler.synthetic import (
    RESULT_REGISTER,
    SyntheticTask,
    build_arithmetic_prompt,
    build_gold_program,
    build_max_program,
    build_max_prompt,
    build_sum_to_n_program,
    build_sum_to_n_prompt,
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

    def to_synthetic_task(self) -> SyntheticTask:
        return SyntheticTask(
            prompt=self.canonical_prompt,
            expected_output=self.expected_output,
            gold_program=self.gold_program,
            result_register=self.result_register,
            task_type=self.task_type,
        )


def generate_nl_tasks(
    *,
    task_types: Sequence[str] = ("arithmetic", "max", "sum_to_n"),
    operations: Sequence[str] = ("add", "sub", "mul", "div"),
    values: Iterable[int] = range(0, 6),
    result_register: int = RESULT_REGISTER,
    seed: int | None = None,
) -> list[NaturalLanguageTask]:
    rng = random.Random(seed)
    tasks: list[NaturalLanguageTask] = []
    cached_values = list(values)
    requested_types = set(task_types)

    if "arithmetic" in requested_types:
        for operation in operations:
            for lhs in cached_values:
                for rhs in cached_values:
                    if operation == "div" and rhs == 0:
                        continue
                    canonical_prompt = build_arithmetic_prompt(operation, lhs, rhs)
                    tasks.append(
                        NaturalLanguageTask(
                            prompt=_arithmetic_template(operation, lhs, rhs, rng=rng),
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
                        )
                    )

    if "max" in requested_types:
        for lhs in cached_values:
            for rhs in cached_values:
                tasks.append(
                    NaturalLanguageTask(
                        prompt=_max_template(lhs, rhs, rng=rng),
                        canonical_prompt=build_max_prompt(lhs, rhs),
                        expected_output=max(lhs, rhs),
                        gold_program=build_max_program(lhs, rhs, result_register=result_register),
                        result_register=result_register,
                        task_type="max",
                    )
                )

    if "sum_to_n" in requested_types:
        for n in cached_values:
            if n < 0:
                raise ValueError("sum_to_n requires a non-negative integer")
            tasks.append(
                NaturalLanguageTask(
                    prompt=_sum_to_n_template(n, rng=rng),
                    canonical_prompt=build_sum_to_n_prompt(n),
                    expected_output=n * (n + 1) // 2,
                    gold_program=build_sum_to_n_program(n, result_register=result_register),
                    result_register=result_register,
                    task_type="sum_to_n",
                )
            )

    return tasks


def _arithmetic_template(operation: str, lhs: int, rhs: int, *, rng: random.Random) -> str:
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
    return rng.choice(templates[operation])


def _max_template(lhs: int, rhs: int, *, rng: random.Random) -> str:
    templates = (
        f"Max of {lhs} and {rhs}",
        f"Which number is larger: {lhs} or {rhs}?",
        f"Compare {lhs} and {rhs} and return the larger",
    )
    return rng.choice(templates)


def _sum_to_n_template(n: int, *, rng: random.Random) -> str:
    templates = (
        f"Sum integers from 1 to {n}",
        f"Sum all integers up to {n}",
        f"Compute the triangular number of {n}",
    )
    return rng.choice(templates)
