from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

from .analysis import ValidationError
from .ir import Instruction
from .machine import Trap, VM
from .state import TraceEntry, VMState


def instruction_to_dict(instruction: Instruction) -> dict[str, Any]:
    return asdict(instruction)


def instruction_from_dict(data: dict[str, Any]) -> Instruction:
    try:
        return Instruction(**data)
    except ValueError as exc:
        raise ValidationError(str(exc)) from exc


def program_to_dict(program: list[Instruction] | tuple[Instruction, ...]) -> dict[str, Any]:
    return {"instructions": [instruction_to_dict(instruction) for instruction in program]}


def program_from_dict(data: dict[str, Any]) -> list[Instruction]:
    raw_instructions = data.get("instructions", [])
    return [instruction_from_dict(entry) for entry in raw_instructions]


def program_to_json(program: list[Instruction] | tuple[Instruction, ...]) -> str:
    return json.dumps(program_to_dict(program), sort_keys=True)


def program_from_json(payload: str) -> list[Instruction]:
    return program_from_dict(json.loads(payload))


def trace_to_dict(trace: list[TraceEntry] | tuple[TraceEntry, ...]) -> dict[str, Any]:
    return {"trace": [asdict(entry) for entry in trace]}


def _normalize_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(snapshot)
    normalized["registers"] = {
        int(key): value for key, value in snapshot.get("registers", {}).items()
    }
    normalized["memory"] = {int(key): value for key, value in snapshot.get("memory", {}).items()}
    normalized["flags"] = dict(snapshot.get("flags", {}))
    normalized["stack"] = list(snapshot.get("stack", []))
    normalized["call_stack"] = list(snapshot.get("call_stack", []))
    return normalized


def trace_from_dict(data: dict[str, Any]) -> list[TraceEntry]:
    entries: list[TraceEntry] = []
    for raw_entry in data.get("trace", []):
        raw_instruction = raw_entry["instruction"]
        entries.append(
            TraceEntry(
                step=raw_entry["step"],
                pc=raw_entry["pc"],
                instruction=instruction_from_dict(raw_instruction),
                pre_state=_normalize_snapshot(raw_entry["pre_state"]),
                post_state=_normalize_snapshot(raw_entry["post_state"]),
                trap=raw_entry.get("trap"),
            )
        )
    return entries


def trace_to_json(trace: list[TraceEntry] | tuple[TraceEntry, ...]) -> str:
    return json.dumps(trace_to_dict(trace), sort_keys=True)


def trace_from_json(payload: str) -> list[TraceEntry]:
    return trace_from_dict(json.loads(payload))


def state_to_dict(state: VMState) -> dict[str, Any]:
    return {
        "registers": state.registers,
        "memory": state.memory,
        "stack": state.stack,
        "call_stack": state.call_stack,
        "pc": state.pc,
        "flags": state.flags,
        "halted": state.halted,
        "halt_reason": state.halt_reason,
        "step_count": state.step_count,
        "trace": trace_to_dict(state.trace)["trace"],
    }


def state_from_dict(data: dict[str, Any]) -> VMState:
    return VMState(
        registers={int(key): value for key, value in data.get("registers", {}).items()},
        memory={int(key): value for key, value in data.get("memory", {}).items()},
        stack=list(data.get("stack", [])),
        call_stack=list(data.get("call_stack", [])),
        pc=data.get("pc", 0),
        flags=dict(data.get("flags", {})),
        halted=data.get("halted", False),
        halt_reason=data.get("halt_reason"),
        step_count=data.get("step_count", 0),
        trace=trace_from_dict({"trace": data.get("trace", [])}),
    )


def state_to_json(state: VMState) -> str:
    return json.dumps(state_to_dict(state), sort_keys=True)


def state_from_json(payload: str) -> VMState:
    return state_from_dict(json.loads(payload))


def trap_to_dict(trap: Trap) -> dict[str, Any]:
    return {
        "kind": trap.kind,
        "pc": trap.pc,
        "instruction": instruction_to_dict(trap.instruction) if trap.instruction is not None else None,
    }


def trap_from_dict(data: dict[str, Any]) -> Trap:
    raw_instruction = data.get("instruction")
    return Trap(
        kind=data["kind"],
        pc=data.get("pc"),
        instruction=instruction_from_dict(raw_instruction) if raw_instruction is not None else None,
    )


def trap_to_json(trap: Trap) -> str:
    return json.dumps(trap_to_dict(trap), sort_keys=True)


def trap_from_json(payload: str) -> Trap:
    return trap_from_dict(json.loads(payload))


def replay_program(
    program_payload: str,
    *,
    state_payload: str | None = None,
    trace: bool = False,
    vm: VM | None = None,
) -> VMState:
    machine = vm if vm is not None else VM()
    program = program_from_json(program_payload)
    state = state_from_json(state_payload) if state_payload is not None else None
    return machine.execute(program, state=state, trace=trace)
