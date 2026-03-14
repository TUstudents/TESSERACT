"""Exact virtual machine and IR."""

from .ir import Instruction, VALID_OPCODES, VALID_TYPE_TAGS
from .machine import Trap, VM
from .state import TraceEntry, VMState

__all__ = [
    "Instruction",
    "VALID_OPCODES",
    "VALID_TYPE_TAGS",
    "TraceEntry",
    "Trap",
    "VM",
    "VMState",
]
