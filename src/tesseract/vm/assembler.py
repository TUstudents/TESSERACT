from __future__ import annotations

from dataclasses import replace

from .analysis import CONTROL_FLOW_OPCODES, ValidationError, validate_program
from .ir import Instruction

AssemblySource = list[str] | tuple[str, ...]


def assemble(source: AssemblySource, *, register_count: int = 32) -> list[Instruction]:
    labels: dict[str, int] = {}
    parsed: list[Instruction] = []

    for line in source:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.endswith(":"):
            label_name = stripped[:-1].strip()
            if not label_name:
                raise ValidationError("empty label definition")
            if label_name in labels:
                raise ValidationError(f"duplicate label {label_name!r}")
            labels[label_name] = len(parsed)
            continue
        parsed.append(_parse_instruction_line(stripped))

    assembled: list[Instruction] = []
    for instruction in parsed:
        if instruction.opcode in CONTROL_FLOW_OPCODES and instruction.label is not None:
            if instruction.label not in labels:
                raise ValidationError(f"undefined label {instruction.label!r}")
            assembled.append(replace(instruction, imm=labels[instruction.label]))
        else:
            assembled.append(instruction)

    validate_program(assembled, register_count=register_count)
    return assembled


def disassemble(program: list[Instruction] | tuple[Instruction, ...]) -> list[str]:
    target_map = _target_labels(program)
    lines: list[str] = []
    for index, instruction in enumerate(program):
        if index in target_map:
            lines.append(f"{target_map[index]}:")
        lines.append(_instruction_to_line(instruction, target_map))
    return lines


def _target_labels(program: list[Instruction] | tuple[Instruction, ...]) -> dict[int, str]:
    targets = sorted(
        {
            instruction.imm
            for instruction in program
            if instruction.opcode in CONTROL_FLOW_OPCODES and isinstance(instruction.imm, int)
        }
    )
    return {target: f"L{slot}" for slot, target in enumerate(targets)}


def _instruction_to_line(instruction: Instruction, target_map: dict[int, str]) -> str:
    fields: list[str] = []
    if instruction.dst is not None:
        fields.append(f"dst={instruction.dst}")
    if instruction.src1 is not None:
        fields.append(f"src1={instruction.src1}")
    if instruction.src2 is not None:
        fields.append(f"src2={instruction.src2}")
    if instruction.opcode in CONTROL_FLOW_OPCODES and isinstance(instruction.imm, int):
        label_name = target_map.get(instruction.imm)
        if label_name is not None:
            fields.append(f"label={label_name}")
        else:
            fields.append(f"imm={instruction.imm}")
    elif instruction.imm is not None:
        fields.append(f"imm={_format_value(instruction.imm)}")
    if instruction.type_tag is not None:
        fields.append(f"type_tag={instruction.type_tag}")
    return " ".join([instruction.opcode, *fields]).strip()


def _parse_instruction_line(line: str) -> Instruction:
    parts = line.split()
    if not parts:
        raise ValidationError("empty instruction line")
    opcode = parts[0]
    dst: int | None = None
    src1: int | None = None
    src2: int | None = None
    imm: int | bool | None = None
    label: str | None = None
    type_tag: str | None = None

    for token in parts[1:]:
        if "=" not in token:
            raise ValidationError(f"malformed operand {token!r}")
        key, raw_value = token.split("=", 1)
        if key == "dst":
            dst = int(raw_value)
        elif key == "src1":
            src1 = int(raw_value)
        elif key == "src2":
            src2 = int(raw_value)
        elif key == "imm":
            imm = _parse_immediate(raw_value)
        elif key == "label":
            label = raw_value
        elif key == "type_tag":
            type_tag = raw_value
        else:
            raise ValidationError(f"unknown operand {key!r}")

    return Instruction(opcode=opcode, dst=dst, src1=src1, src2=src2, imm=imm, label=label, type_tag=type_tag)


def _parse_immediate(raw_value: str) -> int | bool:
    lowered = raw_value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    return int(raw_value)


def _format_value(value: int | bool) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    return str(value)
