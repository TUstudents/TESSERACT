from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from tesseract.vm.state import TraceEntry, VMState

TraceStatus = Literal["success", "failure"]
FailureType = Literal[
    "SUCCESS",
    "WRONG_BRANCH",
    "WRONG_REGISTER",
    "WRONG_ADDRESS",
    "WRONG_VALUE",
    "TYPE_ERROR",
    "TIMEOUT",
    "INVALID_OP",
    "INVARIANT_VIOLATION",
    "UNKNOWN_FAILURE",
]


@dataclass(frozen=True)
class TraceSummary:
    status: TraceStatus
    halt_reason: str | None
    step_count: int
    final_pc: int
    trap: str | None
    final_registers: dict[int, int | bool]
    final_memory: dict[int, int | bool]


@dataclass(frozen=True)
class InvariantViolation:
    name: str
    message: str
    step: int | None = None


@dataclass(frozen=True)
class CriticReport:
    status: TraceStatus
    failure_type: FailureType
    first_failing_step: int | None
    message: str
    candidate_summary: TraceSummary
    expected_summary: TraceSummary | None = None
    differing_registers: tuple[int, ...] = ()
    differing_addresses: tuple[int, ...] = ()
    invariant_violations: tuple[InvariantViolation, ...] = ()
    repair_prompt: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def summarize_trace(state: VMState) -> TraceSummary:
    trap = None
    if state.trace and state.trace[-1].trap is not None:
        trap = state.trace[-1].trap
    status: TraceStatus = "success" if trap is None and state.halt_reason == "HALT" else "failure"
    return TraceSummary(
        status=status,
        halt_reason=state.halt_reason,
        step_count=state.step_count,
        final_pc=state.pc,
        trap=trap,
        final_registers=dict(state.registers),
        final_memory=dict(state.memory),
    )


def coerce_trace_entries(trace: VMState | list[TraceEntry] | tuple[TraceEntry, ...]) -> list[TraceEntry]:
    if isinstance(trace, VMState):
        return list(trace.trace)
    if not isinstance(trace, (list, tuple)):
        raise TypeError("trace must be a VMState or a list/tuple of TraceEntry objects")
    entries = list(trace)
    for index, entry in enumerate(entries):
        if not isinstance(entry, TraceEntry):
            raise TypeError(f"trace entry at index {index} is not a TraceEntry")
    return entries
