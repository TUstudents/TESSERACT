from __future__ import annotations

from dataclasses import dataclass
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
SUPPORTED_TASK_TYPES = frozenset({
    "arithmetic",
    "max",
    "sum_to_n",
    "factorial",
    "fibonacci",
    "abs",
    "memory_sum",
})
RESULT_REGISTER = 2


def _allocate_temp_registers(*, excluded: set[int], count: int, register_count: int = 32) -> tuple[int, ...]:
    registers = tuple(register for register in range(register_count) if register not in excluded)
    if len(registers) < count:
        raise ValueError("not enough registers available for synthetic program construction")
    return registers[:count]


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
    values: tuple[int, ...] = ()


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


def build_factorial_prompt(n: int) -> str:
    return f"factorial {n}"


def build_fibonacci_prompt(n: int) -> str:
    return f"fibonacci {n}"


def build_abs_prompt(value: int) -> str:
    return f"abs {value}"


def build_memory_sum_prompt(values: Sequence[int]) -> str:
    if not values:
        return "memory_sum"
    return "memory_sum " + " ".join(str(value) for value in values)


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
    program = tuple(
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
    validate_program(program)
    return program


def build_sum_to_n_program(n: int, *, result_register: int = RESULT_REGISTER) -> tuple[Instruction, ...]:
    if n < 0:
        raise ValueError("sum_to_n requires a non-negative integer")
    limit_register, counter_register, one_register, compare_register = _allocate_temp_registers(
        excluded={result_register},
        count=4,
    )
    program = tuple(
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
    validate_program(program)
    return program


def build_factorial_program(n: int, *, result_register: int = RESULT_REGISTER) -> tuple[Instruction, ...]:
    if n < 0:
        raise ValueError("factorial requires a non-negative integer")
    limit_register, counter_register, one_register, compare_register = _allocate_temp_registers(
        excluded={result_register},
        count=4,
    )
    program = tuple(
        assemble(
            [
                f"CONST dst={limit_register} imm={n}",
                f"CONST dst={result_register} imm=1",
                f"CONST dst={counter_register} imm=1",
                f"CONST dst={one_register} imm=1",
                "loop:",
                f"CMP_GT dst={compare_register} src1={counter_register} src2={limit_register}",
                "JGT label=done",
                f"MUL dst={result_register} src1={result_register} src2={counter_register}",
                f"ADD dst={counter_register} src1={counter_register} src2={one_register}",
                "JMP label=loop",
                "done:",
                "HALT",
            ]
        )
    )
    validate_program(program)
    return program


def build_fibonacci_program(n: int, *, result_register: int = RESULT_REGISTER) -> tuple[Instruction, ...]:
    if n < 0:
        raise ValueError("fibonacci requires a non-negative integer")
    zero_register, one_register, limit_register, prev_register, curr_register, next_register, index_register, compare_register = _allocate_temp_registers(
        excluded={result_register},
        count=8,
    )
    program = tuple(
        assemble(
            [
                f"CONST dst={limit_register} imm={n}",
                f"CONST dst={result_register} imm=0",
                f"CONST dst={zero_register} imm=0",
                f"CONST dst={one_register} imm=1",
                f"CONST dst={prev_register} imm=0",
                f"CONST dst={curr_register} imm=1",
                f"CMP_EQ dst={compare_register} src1={limit_register} src2={zero_register}",
                f"JNZ src1={compare_register} label=done",
                f"CMP_EQ dst={compare_register} src1={limit_register} src2={one_register}",
                f"JNZ src1={compare_register} label=return_one",
                f"CONST dst={index_register} imm=2",
                "loop:",
                f"CMP_GT dst={compare_register} src1={index_register} src2={limit_register}",
                "JGT label=return_curr",
                f"ADD dst={next_register} src1={prev_register} src2={curr_register}",
                f"MOV dst={prev_register} src1={curr_register}",
                f"MOV dst={curr_register} src1={next_register}",
                f"ADD dst={index_register} src1={index_register} src2={one_register}",
                "JMP label=loop",
                "return_one:",
                f"MOV dst={result_register} src1={one_register}",
                "JMP label=done",
                "return_curr:",
                f"MOV dst={result_register} src1={curr_register}",
                "done:",
                "HALT",
            ]
        )
    )
    validate_program(program)
    return program


def build_abs_program(value: int, *, result_register: int = RESULT_REGISTER) -> tuple[Instruction, ...]:
    input_register, zero_register, negative_one_register, compare_register = _allocate_temp_registers(
        excluded={result_register},
        count=4,
    )
    program = tuple(
        assemble(
            [
                f"CONST dst={input_register} imm={value}",
                f"CONST dst={zero_register} imm=0",
                f"CONST dst={negative_one_register} imm=-1",
                f"CMP_LT dst={compare_register} src1={input_register} src2={zero_register}",
                f"JNZ src1={compare_register} label=negate",
                f"MOV dst={result_register} src1={input_register}",
                "JMP label=done",
                "negate:",
                f"MUL dst={result_register} src1={input_register} src2={negative_one_register}",
                "done:",
                "HALT",
            ]
        )
    )
    validate_program(program)
    return program


def build_memory_sum_program(values: Sequence[int], *, result_register: int = RESULT_REGISTER) -> tuple[Instruction, ...]:
    base_register, preload_register, address_register, last_index_register, one_register, value_register, compare_register = _allocate_temp_registers(
        excluded={result_register},
        count=7,
    )
    instructions = [
        f"CONST dst={base_register} imm=0",
    ]
    for index, value in enumerate(values):
        instructions.extend(
            [
                f"CONST dst={preload_register} imm={value}",
                f"STORE src1={base_register} src2={preload_register} imm={index}",
            ]
        )
    instructions.extend(
        [
            f"CONST dst={address_register} imm=0",
            f"CONST dst={last_index_register} imm={len(values) - 1}",
            f"CONST dst={result_register} imm=0",
            f"CONST dst={one_register} imm=1",
            "loop:",
            f"CMP_GT dst={compare_register} src1={address_register} src2={last_index_register}",
            "JGT label=done",
            f"LOAD dst={value_register} src1={address_register}",
            f"ADD dst={result_register} src1={result_register} src2={value_register}",
            f"ADD dst={address_register} src1={address_register} src2={one_register}",
            "JMP label=loop",
            "done:",
            "HALT",
        ]
    )
    program = tuple(assemble(instructions))
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


def evaluate_factorial(n: int) -> int:
    if n < 0:
        raise ValueError("factorial requires a non-negative integer")
    result = 1
    for value in range(1, n + 1):
        result *= value
    return result


def evaluate_fibonacci(n: int) -> int:
    if n < 0:
        raise ValueError("fibonacci requires a non-negative integer")
    if n == 0:
        return 0
    prev, curr = 0, 1
    for _ in range(2, n + 1):
        prev, curr = curr, prev + curr
    return curr


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
        values=(lhs, rhs),
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
        values=(lhs, rhs),
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
        values=(n,),
    )


def make_factorial_task(
    n: int,
    *,
    result_register: int = RESULT_REGISTER,
) -> SyntheticTask:
    return SyntheticTask(
        prompt=build_factorial_prompt(n),
        expected_output=evaluate_factorial(n),
        gold_program=build_factorial_program(n, result_register=result_register),
        result_register=result_register,
        task_type="factorial",
        n=n,
        values=(n,),
    )


def make_fibonacci_task(
    n: int,
    *,
    result_register: int = RESULT_REGISTER,
) -> SyntheticTask:
    return SyntheticTask(
        prompt=build_fibonacci_prompt(n),
        expected_output=evaluate_fibonacci(n),
        gold_program=build_fibonacci_program(n, result_register=result_register),
        result_register=result_register,
        task_type="fibonacci",
        n=n,
        values=(n,),
    )


def make_abs_task(
    value: int,
    *,
    result_register: int = RESULT_REGISTER,
) -> SyntheticTask:
    return SyntheticTask(
        prompt=build_abs_prompt(value),
        expected_output=abs(value),
        gold_program=build_abs_program(value, result_register=result_register),
        result_register=result_register,
        task_type="abs",
        n=value,
        values=(value,),
    )


def make_memory_sum_task(
    values: Sequence[int],
    *,
    result_register: int = RESULT_REGISTER,
) -> SyntheticTask:
    sequence = tuple(values)
    return SyntheticTask(
        prompt=build_memory_sum_prompt(sequence),
        expected_output=sum(sequence),
        gold_program=build_memory_sum_program(sequence, result_register=result_register),
        result_register=result_register,
        task_type="memory_sum",
        values=sequence,
    )


def generate_synthetic_tasks(
    *,
    task_types: Sequence[str] = ("arithmetic", "max", "sum_to_n"),
    operations: Sequence[str] = ("add", "sub", "mul", "div"),
    values: Iterable[int] = range(0, 6),
    result_register: int = RESULT_REGISTER,
    memory_sequences: Sequence[Sequence[int]] | None = None,
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

    if "factorial" in requested_types:
        for n in cached_values:
            tasks.append(make_factorial_task(n, result_register=result_register))

    if "fibonacci" in requested_types:
        for n in cached_values:
            tasks.append(make_fibonacci_task(n, result_register=result_register))

    if "abs" in requested_types:
        signed_values = sorted({*cached_values, *(-value for value in cached_values)})
        for value in signed_values:
            tasks.append(make_abs_task(value, result_register=result_register))

    if "memory_sum" in requested_types:
        sequences = memory_sequences if memory_sequences is not None else _default_memory_sequences(cached_values)
        for sequence in sequences:
            tasks.append(make_memory_sum_task(sequence, result_register=result_register))

    return tasks


def execute_task(task: SyntheticTask, vm: VM | None = None) -> TaskExecutionResult:
    machine = vm if vm is not None else VM()
    state = machine.execute(task.gold_program)
    return TaskExecutionResult(
        output=state.registers[task.result_register],
        program_length=len(task.gold_program),
        result_register=task.result_register,
    )
