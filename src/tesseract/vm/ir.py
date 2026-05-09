from __future__ import annotations

from dataclasses import dataclass
from typing import Final

VMImmediate = int | bool | float

VALID_OPCODES: Final[frozenset[str]] = frozenset(
    {
        "MOV",
        "CONST",
        "ADD",
        "SUB",
        "MUL",
        "DIV",
        "AND",
        "OR",
        "NOT",
        "XOR",
        "CMP_EQ",
        "CMP_LT",
        "CMP_GT",
        "JMP",
        "JZ",
        "JNZ",
        "JLT",
        "JGT",
        "LOAD",
        "STORE",
        "PUSH",
        "POP",
        "CALL",
        "RET",
        "HALT",
    }
)

VALID_TYPE_TAGS: Final[frozenset[str]] = frozenset(
    {
        "bool",
        "int",
        "i32",
        "i64",
        "checked_i32",
        "checked_i64",
        "f32",
        "addr",
    }
)


@dataclass(frozen=True)
class Instruction:
    opcode: str
    dst: int | None = None
    src1: int | None = None
    src2: int | None = None
    imm: VMImmediate | None = None
    label: str | None = None
    type_tag: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "opcode", self.opcode.upper())
        if self.opcode not in VALID_OPCODES:
            raise ValueError(f"unknown opcode {self.opcode!r}")
