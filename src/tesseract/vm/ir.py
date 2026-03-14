from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Instruction:
    opcode: str
    dst: Optional[int] = None
    src1: Optional[int] = None
    src2: Optional[int] = None
    imm: Optional[int] = None
    label: Optional[str] = None
    type_tag: Optional[str] = None
