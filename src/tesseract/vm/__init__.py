"""Exact virtual machine and IR."""

from .analysis import TypeCheckResult, ValidationError, validate_program
from .assembler import assemble, disassemble
from .ir import Instruction, VALID_OPCODES, VALID_TYPE_TAGS
from .machine import Trap, VM
from .serialization import (
    instruction_from_dict,
    instruction_to_dict,
    program_from_dict,
    program_from_json,
    program_to_dict,
    program_to_json,
    replay_program,
    state_from_dict,
    state_from_json,
    state_to_dict,
    state_to_json,
    trace_from_dict,
    trace_from_json,
    trace_to_dict,
    trace_to_json,
    trap_from_dict,
    trap_from_json,
    trap_to_dict,
    trap_to_json,
)
from .state import TraceEntry, VMState

__all__ = [
    "Instruction",
    "VALID_OPCODES",
    "VALID_TYPE_TAGS",
    "TypeCheckResult",
    "ValidationError",
    "validate_program",
    "assemble",
    "disassemble",
    "instruction_to_dict",
    "instruction_from_dict",
    "program_to_dict",
    "program_from_dict",
    "program_to_json",
    "program_from_json",
    "state_to_dict",
    "state_from_dict",
    "state_to_json",
    "state_from_json",
    "trace_to_dict",
    "trace_from_dict",
    "trace_to_json",
    "trace_from_json",
    "trap_to_dict",
    "trap_from_dict",
    "trap_to_json",
    "trap_from_json",
    "replay_program",
    "TraceEntry",
    "Trap",
    "VM",
    "VMState",
]
