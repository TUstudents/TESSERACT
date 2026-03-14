from dataclasses import dataclass
from typing import Iterable, List

from .ir import Instruction
from .state import VMState


@dataclass
class Trap(Exception):
    kind: str


class VM:
    def __init__(self, step_budget: int = 10_000):
        self.step_budget = step_budget

    def execute(self, program: Iterable[Instruction], state: VMState | None = None) -> VMState:
        prog: List[Instruction] = list(program)
        s = state or VMState()
        steps = 0
        while 0 <= s.pc < len(prog):
            if steps >= self.step_budget:
                raise Trap("TIMEOUT")
            ins = prog[s.pc]
            if ins.opcode == "HALT":
                return s
            elif ins.opcode == "CONST":
                assert ins.dst is not None
                assert ins.imm is not None
                s.registers[ins.dst] = ins.imm
                s.pc += 1
            elif ins.opcode == "ADD":
                assert ins.dst is not None
                assert ins.src1 is not None
                assert ins.src2 is not None
                s.registers[ins.dst] = s.registers.get(ins.src1, 0) + s.registers.get(ins.src2, 0)
                s.pc += 1
            else:
                raise Trap(f"INVALID_OP:{ins.opcode}")
            steps += 1
        return s
