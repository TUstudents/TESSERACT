from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .ir import Instruction

VMValue = int | bool


@dataclass(frozen=True)
class TraceEntry:
    step: int
    pc: int
    instruction: Instruction
    pre_state: dict[str, Any]
    post_state: dict[str, Any]
    trap: str | None = None


@dataclass
class VMState:
    registers: dict[int, VMValue] = field(default_factory=dict)
    memory: dict[int, VMValue] = field(default_factory=dict)
    stack: list[VMValue] = field(default_factory=list)
    call_stack: list[int] = field(default_factory=list)
    pc: int = 0
    flags: dict[str, bool] = field(
        default_factory=lambda: {"zero": False, "lt": False, "gt": False, "eq": False}
    )
    halted: bool = False
    halt_reason: str | None = None
    step_count: int = 0
    trace: list[TraceEntry] = field(default_factory=list)

    def snapshot(self) -> dict[str, Any]:
        return {
            "registers": deepcopy(self.registers),
            "memory": deepcopy(self.memory),
            "stack": list(self.stack),
            "call_stack": list(self.call_stack),
            "pc": self.pc,
            "flags": dict(self.flags),
            "halted": self.halted,
            "halt_reason": self.halt_reason,
            "step_count": self.step_count,
        }
