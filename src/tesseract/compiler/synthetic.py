from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import product
from typing import Iterable, Sequence

from tesseract.vm import Instruction, VM, assemble, validate_program

VM_OPCODE_BY_TASK = {
    "add": "ADD",
    "sub": "SUB",
    "mul": "MUL",
    "div": "DIV",
}
SUPPORTED_OPERATIONS = frozenset(VM_OPCODE_BY_TASK)
SUPPORTED_TASK_TYPES = frozenset({"arithmetic", "max", "sum_to_n"})
RESULT_REGISTER = 2


def _strip_instruction_labels(program: tuple[Instruction, ...]) -> tuple[Instruction, ...]:
    return tuple(replace(instruction, label=None) for instruction in program)


def _allocate_temp_registers(*, excluded: set[int], count: int, register_count: int = 32) -> tuple[int, ...]:
    registers = tuple(register for register in range(register_count) if register not in excluded)
    if len(registers) < count:
        raise ValueError("not enough registers available for synthetic program construction")
    return registers[:count]


@dataclass(frozen=True)
class SyntheticTask:
    prompt: str
    expected_output: int
    gold_program: tuple[Instruction, ...]
    result_register: int = RESULT_REGISTER
    task_type: str = "arithmetic"
    operation: str | None = None
    lhs: int | None = None
    rhs: int | None = None
    n: int | None = None


@dataclass(frozen=True)
class TaskExecutionResult:
    output: int
    program_length: int
    result_register: int


def build_arithmetic_prompt(operation: str, lhs: int, rhs: int) -> str:
    return f"arith {operation} {lhs} {rhs}"


def build_max_prompt(lhs: int, rhs: int) -> str:
    return f"max {lhs} {rhs}"


def build_sum_to_n_prompt(n: int) -> str:
    return f"sum_to_n {n}"


def build_gold_program(
    operation: str,
    lhs: int,
    rhs: int,
    *,
    result_register: int = RESULT_REGISTER,
) -> tuple[Instruction, ...]:
    if operation not in VM_OPCODE_BY_TASK:
        raise ValueError(f"unsupported operation {operation!r}")
    opcode = VM_OPCODE_BY_TASK[operation]
    program = (
        Instruction("CONST", dst=0, imm=lhs),
        Instruction("CONST", dst=1, imm=rhs),
        Instruction(opcode, dst=result_register, src1=0, src2=1),
        Instruction("HALT"),
    )
    validate_program(program)
    return program


def build_max_program(lhs: int, rhs: int, *, result_register: int = RESULT_REGISTER) -> tuple[Instruction, ...]:
    program = _strip_instruction_labels(
        tuple(
            assemble(
                [
                    f"CONST dst=0 imm={lhs}",
                    f"CONST dst=1 imm={rhs}",
                    "CMP_GT dst=3 src1=0 src2=1",
                    "JGT label=lhs_wins",
                    f"MOV dst={result_register} src1=1",
                    "JMP label=done",
                    "lhs_wins:",
                    f"MOV dst={result_register} src1=0",
                    "done:",
                    "HALT",
                ]
            )
        )
    )
    validate_program(program)
    return program


def build_sum_to_n_program(n: int, *, result_register: int = RESULT_REGISTER) -> tuple[Instruction, ...]:
    if n < 0:
        raise ValueError("sum_to_n requires a non-negative integer")
    limit_register, counter_register, one_register, compare_register = _allocate_temp_registers(
        excluded={result_register},
        count=4,
    )
    program = _strip_instruction_labels(
        tuple(
            assemble(
                [
                    f"CONST dst={limit_register} imm={n}",
                    f"CONST dst={result_register} imm=0",
                    f"CONST dst={counter_register} imm=1",
                    f"CONST dst={one_register} imm=1",
                    "loop:",
                    f"CMP_GT dst={compare_register} src1={counter_register} src2={limit_register}",
                    "JGT label=done",
                    f"ADD dst={result_register} src1={result_register} src2={counter_register}",
                    f"ADD dst={counter_register} src1={counter_register} src2={one_register}",
                    "JMP label=loop",
                    "done:",
                    "HALT",
                ]
            )
        )
    )
    validate_program(program)
    return program


def evaluate_operation(operation: str, lhs: int, rhs: int) -> int:
    if operation == "add":
        return lhs + rhs
    if operation == "sub":
        return lhs - rhs
    if operation == "mul":
        return lhs * rhs
    if operation == "div":
        return int(lhs / rhs)
    raise ValueError(f"unsupported operation {operation!r}")


def make_synthetic_task(
    operation: str,
    lhs: int,
    rhs: int,
    *,
    result_register: int = RESULT_REGISTER,
) -> SyntheticTask:
    return SyntheticTask(
        prompt=build_arithmetic_prompt(operation, lhs, rhs),
        expected_output=evaluate_operation(operation, lhs, rhs),
        gold_program=build_gold_program(operation, lhs, rhs, result_register=result_register),
        result_register=result_register,
        task_type="arithmetic",
        operation=operation,
        lhs=lhs,
        rhs=rhs,
    )


def make_max_task(
    lhs: int,
    rhs: int,
    *,
    result_register: int = RESULT_REGISTER,
) -> SyntheticTask:
    return SyntheticTask(
        prompt=build_max_prompt(lhs, rhs),
        expected_output=max(lhs, rhs),
        gold_program=build_max_program(lhs, rhs, result_register=result_register),
        result_register=result_register,
        task_type="max",
        lhs=lhs,
        rhs=rhs,
    )


def make_sum_to_n_task(
    n: int,
    *,
    result_register: int = RESULT_REGISTER,
) -> SyntheticTask:
    if n < 0:
        raise ValueError("sum_to_n requires a non-negative integer")
    return SyntheticTask(
        prompt=build_sum_to_n_prompt(n),
        expected_output=n * (n + 1) // 2,
        gold_program=build_sum_to_n_program(n, result_register=result_register),
        result_register=result_register,
        task_type="sum_to_n",
        n=n,
    )


def generate_synthetic_tasks(
    *,
    task_types: Sequence[str] = ("arithmetic", "max", "sum_to_n"),
    operations: Sequence[str] = ("add", "sub", "mul", "div"),
    values: Iterable[int] = range(0, 6),
    result_register: int = RESULT_REGISTER,
) -> list[SyntheticTask]:
    tasks: list[SyntheticTask] = []
    cached_values = list(values)
    requested_types = set(task_types)
    unknown_task_types = requested_types - SUPPORTED_TASK_TYPES
    if unknown_task_types:
        raise ValueError(f"unsupported task type(s): {sorted(unknown_task_types)!r}")
    unknown_operations = set(operations) - SUPPORTED_OPERATIONS
    if unknown_operations:
        raise ValueError(f"unsupported operation(s): {sorted(unknown_operations)!r}")

    if "arithmetic" in requested_types:
        for operation, lhs, rhs in product(operations, cached_values, cached_values):
            if operation == "div" and rhs == 0:
                continue
            tasks.append(make_synthetic_task(operation, lhs, rhs, result_register=result_register))

    if "max" in requested_types:
        for lhs, rhs in product(cached_values, cached_values):
            tasks.append(make_max_task(lhs, rhs, result_register=result_register))

    if "sum_to_n" in requested_types:
        for n in cached_values:
            tasks.append(make_sum_to_n_task(n, result_register=result_register))

    return tasks


def execute_task(task: SyntheticTask, vm: VM | None = None) -> TaskExecutionResult:
    machine = vm if vm is not None else VM()
    state = machine.execute(task.gold_program)
    return TaskExecutionResult(
        output=state.registers[task.result_register],
        program_length=len(task.gold_program),
        result_register=task.result_register,
    )
