from __future__ import annotations

from dataclasses import dataclass
from typing import Collection, Final, Literal, cast

from .ir import Instruction, VALID_OPCODES, VALID_TYPE_TAGS

RegisterType = Literal["bool", "int", "i32", "i64", "checked_i32"]
CONTROL_FLOW_OPCODES: Final[frozenset[str]] = frozenset({"JMP", "JZ", "JNZ", "JLT", "JGT", "CALL"})


class ValidationError(Exception):
    def __init__(self, message: str, index: int | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.index = index

    def __str__(self) -> str:
        if self.index is None:
            return self.message
        return f"instruction {self.index}: {self.message}"


@dataclass(frozen=True)
class TypeCheckResult:
    register_types: dict[int, RegisterType]


INT_TYPE_TAGS: Final[frozenset[RegisterType]] = frozenset({"int", "i32", "i64", "checked_i32"})
BOOL_TYPE_TAGS: Final[frozenset[RegisterType]] = frozenset({"bool"})


def validate_program(
    program: list[Instruction] | tuple[Instruction, ...],
    *,
    register_count: int = 32,
    allow_unresolved_labels: bool = False,
    require_terminal_halt: bool = True,
) -> TypeCheckResult:
    register_types: dict[int, RegisterType] = {}

    if require_terminal_halt:
        if not program:
            raise ValidationError("program must not be empty")
        if program[-1].opcode != "HALT":
            raise ValidationError("program must terminate with HALT", index=len(program) - 1)

    for index, instruction in enumerate(program):
        _validate_instruction_shape(
            instruction,
            index=index,
            register_count=register_count,
            allow_unresolved_labels=allow_unresolved_labels,
        )
        _update_type_environment(register_types, instruction, index=index)

    return TypeCheckResult(register_types=dict(register_types))


def _validate_instruction_shape(
    instruction: Instruction,
    *,
    index: int,
    register_count: int,
    allow_unresolved_labels: bool,
) -> None:
    opcode = instruction.opcode
    if opcode not in VALID_OPCODES:
        raise ValidationError(f"invalid opcode {opcode!r}", index=index)
    if instruction.type_tag is not None and instruction.type_tag not in VALID_TYPE_TAGS:
        raise ValidationError(f"invalid type tag {instruction.type_tag!r}", index=index)
    if instruction.label is not None and opcode not in CONTROL_FLOW_OPCODES:
        raise ValidationError("labels are only valid on control-flow instructions", index=index)

    def ensure_register(name: str, value: int | None, *, required: bool) -> None:
        if value is None:
            if required:
                raise ValidationError(f"missing register operand {name}", index=index)
            return
        if not 0 <= value < register_count:
            raise ValidationError(f"register {name}={value} out of range", index=index)

    def ensure_no_register(name: str, value: int | None) -> None:
        if value is not None:
            raise ValidationError(f"unexpected register operand {name}", index=index)

    def ensure_target(*, required: bool) -> None:
        has_imm = isinstance(instruction.imm, int) and not isinstance(instruction.imm, bool)
        has_label = instruction.label is not None
        if required and not has_imm and not has_label:
            raise ValidationError("missing branch target", index=index)
        if has_label and not allow_unresolved_labels and not has_imm:
            raise ValidationError("unresolved label target", index=index)

    if opcode in {"HALT", "RET"}:
        ensure_no_register("dst", instruction.dst)
        ensure_no_register("src1", instruction.src1)
        ensure_no_register("src2", instruction.src2)
        if instruction.imm is not None:
            raise ValidationError("unexpected immediate operand", index=index)
        return

    if opcode == "CONST":
        ensure_register("dst", instruction.dst, required=True)
        ensure_no_register("src1", instruction.src1)
        ensure_no_register("src2", instruction.src2)
        if instruction.imm is None:
            raise ValidationError("missing immediate operand", index=index)
        if isinstance(instruction.imm, bool) and instruction.type_tag != "bool":
            raise ValidationError("boolean immediate requires bool type tag", index=index)
        if isinstance(instruction.imm, int) and not isinstance(instruction.imm, bool):
            return
        if isinstance(instruction.imm, bool):
            return
        raise ValidationError("unsupported immediate type", index=index)

    if opcode == "MOV":
        ensure_register("dst", instruction.dst, required=True)
        ensure_register("src1", instruction.src1, required=True)
        ensure_no_register("src2", instruction.src2)
        if instruction.imm is not None:
            raise ValidationError("unexpected immediate operand", index=index)
        return

    if opcode in {"ADD", "SUB", "MUL", "DIV", "AND", "OR", "XOR", "CMP_EQ", "CMP_LT", "CMP_GT"}:
        ensure_register("dst", instruction.dst, required=True)
        ensure_register("src1", instruction.src1, required=True)
        ensure_register("src2", instruction.src2, required=True)
        if instruction.imm is not None:
            raise ValidationError("unexpected immediate operand", index=index)
        return

    if opcode == "NOT":
        ensure_register("dst", instruction.dst, required=True)
        ensure_register("src1", instruction.src1, required=True)
        ensure_no_register("src2", instruction.src2)
        if instruction.imm is not None:
            raise ValidationError("unexpected immediate operand", index=index)
        return

    if opcode in {"JMP", "CALL", "JLT", "JGT"}:
        ensure_no_register("dst", instruction.dst)
        ensure_no_register("src1", instruction.src1)
        ensure_no_register("src2", instruction.src2)
        ensure_target(required=True)
        return

    if opcode in {"JZ", "JNZ"}:
        ensure_no_register("dst", instruction.dst)
        ensure_register("src1", instruction.src1, required=True)
        ensure_no_register("src2", instruction.src2)
        ensure_target(required=True)
        return

    if opcode == "LOAD":
        ensure_register("dst", instruction.dst, required=True)
        ensure_register("src1", instruction.src1, required=True)
        ensure_no_register("src2", instruction.src2)
        if instruction.imm is not None and type(instruction.imm) is not int:
            raise ValidationError("load offset must be an integer", index=index)
        return

    if opcode == "STORE":
        ensure_no_register("dst", instruction.dst)
        ensure_register("src1", instruction.src1, required=True)
        ensure_register("src2", instruction.src2, required=True)
        if instruction.imm is not None and type(instruction.imm) is not int:
            raise ValidationError("store offset must be an integer", index=index)
        return

    if opcode == "PUSH":
        ensure_no_register("dst", instruction.dst)
        ensure_register("src1", instruction.src1, required=True)
        ensure_no_register("src2", instruction.src2)
        if instruction.imm is not None:
            raise ValidationError("unexpected immediate operand", index=index)
        return

    if opcode == "POP":
        ensure_register("dst", instruction.dst, required=True)
        ensure_no_register("src1", instruction.src1)
        ensure_no_register("src2", instruction.src2)
        if instruction.imm is not None:
            raise ValidationError("unexpected immediate operand", index=index)
        return


def _update_type_environment(
    register_types: dict[int, RegisterType],
    instruction: Instruction,
    *,
    index: int,
) -> None:
    opcode = instruction.opcode

    def require_register(register: int | None, name: str) -> int:
        if register is None:
            raise ValidationError(f"missing register operand {name}", index=index)
        return register

    def expect_type(register: int | None, allowed: Collection[RegisterType], name: str) -> RegisterType | None:
        resolved_register = require_register(register, name)
        known = register_types.get(resolved_register)
        if known is not None and known not in allowed:
            raise ValidationError(
                f"register r{resolved_register} has type {known!r}, expected one of {sorted(allowed)!r}",
                index=index,
            )
        return known

    if opcode == "CONST":
        dst = require_register(instruction.dst, "dst")
        if isinstance(instruction.imm, bool):
            register_types[dst] = "bool"
        elif instruction.type_tag in INT_TYPE_TAGS:
            register_types[dst] = cast(RegisterType, instruction.type_tag)
        else:
            register_types[dst] = "int"
        return

    if opcode == "MOV":
        dst = require_register(instruction.dst, "dst")
        src1 = require_register(instruction.src1, "src1")
        source_type = register_types.get(src1)
        if instruction.type_tag == "bool":
            if source_type is not None and source_type != "bool":
                raise ValidationError("cannot move non-bool into bool destination", index=index)
            register_types[dst] = "bool"
        elif instruction.type_tag in INT_TYPE_TAGS:
            if source_type is not None and source_type == "bool":
                raise ValidationError("cannot move bool into integer destination", index=index)
            register_types[dst] = cast(RegisterType, instruction.type_tag)
        elif source_type is not None:
            register_types[dst] = source_type
        return

    if opcode in {"ADD", "SUB", "MUL", "DIV", "CMP_LT", "CMP_GT"}:
        expect_type(instruction.src1, INT_TYPE_TAGS, "src1")
        expect_type(instruction.src2, INT_TYPE_TAGS, "src2")
        if instruction.dst is not None:
            if opcode in {"CMP_LT", "CMP_GT"}:
                register_types[instruction.dst] = "bool"
            else:
                register_types[instruction.dst] = (
                    cast(RegisterType, instruction.type_tag)
                    if instruction.type_tag in INT_TYPE_TAGS
                    else "int"
                )
        return

    if opcode in {"AND", "OR", "XOR", "NOT"}:
        expect_type(instruction.src1, BOOL_TYPE_TAGS, "src1")
        if opcode != "NOT":
            expect_type(instruction.src2, BOOL_TYPE_TAGS, "src2")
        if instruction.dst is not None:
            register_types[instruction.dst] = "bool"
        return

    if opcode == "CMP_EQ":
        dst = require_register(instruction.dst, "dst")
        src1 = require_register(instruction.src1, "src1")
        src2 = require_register(instruction.src2, "src2")
        lhs = register_types.get(src1)
        rhs = register_types.get(src2)
        if lhs is not None and rhs is not None and lhs != rhs:
            raise ValidationError("CMP_EQ operands must have matching static types", index=index)
        register_types[dst] = "bool"
        return

    if opcode == "LOAD":
        expect_type(instruction.src1, INT_TYPE_TAGS, "src1")
        dst = require_register(instruction.dst, "dst")
        if instruction.type_tag == "bool":
            register_types[dst] = "bool"
        elif instruction.type_tag in INT_TYPE_TAGS:
            register_types[dst] = cast(RegisterType, instruction.type_tag)
        return

    if opcode == "STORE":
        expect_type(instruction.src1, INT_TYPE_TAGS, "src1")
        src2 = require_register(instruction.src2, "src2")
        source_type = register_types.get(src2)
        if instruction.type_tag == "bool" and source_type is not None and source_type != "bool":
            raise ValidationError("STORE type tag bool conflicts with integer source", index=index)
        if instruction.type_tag in INT_TYPE_TAGS and source_type == "bool":
            raise ValidationError("STORE integer type tag conflicts with bool source", index=index)
        return

    if opcode in {"PUSH", "POP", "JMP", "JLT", "JGT", "CALL", "RET", "HALT", "JZ", "JNZ"}:
        return
