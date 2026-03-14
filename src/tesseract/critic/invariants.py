from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

from .schema import InvariantViolation
from tesseract.vm.state import TraceEntry, VMState


class Invariant(Protocol):
    name: str

    def check(self, state: VMState) -> InvariantViolation | None:
        ...


@dataclass(frozen=True)
class NoTrapInvariant:
    name: str = "no_trap"

    def check(self, state: VMState) -> InvariantViolation | None:
        if state.trace and state.trace[-1].trap is not None:
            return InvariantViolation(
                name=self.name,
                message=f"execution trapped with {state.trace[-1].trap}",
                step=state.trace[-1].step,
            )
        if state.halt_reason not in {None, "HALT"}:
            return InvariantViolation(name=self.name, message=f"execution halted with {state.halt_reason}")
        return None


@dataclass(frozen=True)
class FinalRegisterInvariant:
    register: int
    expected: int | bool
    name: str = "final_register"

    def check(self, state: VMState) -> InvariantViolation | None:
        observed = state.registers.get(self.register, 0)
        if observed != self.expected:
            return InvariantViolation(
                name=self.name,
                message=f"expected r{self.register}={self.expected!r}, observed {observed!r}",
            )
        return None


@dataclass(frozen=True)
class FinalMemoryInvariant:
    address: int
    expected: int | bool
    name: str = "final_memory"

    def check(self, state: VMState) -> InvariantViolation | None:
        observed = state.memory.get(self.address, 0)
        if observed != self.expected:
            return InvariantViolation(
                name=self.name,
                message=f"expected mem[{self.address}]={self.expected!r}, observed {observed!r}",
            )
        return None


@dataclass(frozen=True)
class MaxStepsInvariant:
    max_steps: int
    name: str = "max_steps"

    def check(self, state: VMState) -> InvariantViolation | None:
        if state.step_count > self.max_steps:
            return InvariantViolation(
                name=self.name,
                message=f"expected at most {self.max_steps} steps, observed {state.step_count}",
            )
        return None


@dataclass(frozen=True)
class TraceStepInvariant:
    step: int
    register: int
    expected: int | bool
    name: str = "trace_step_register"

    def check(self, state: VMState) -> InvariantViolation | None:
        if self.step >= len(state.trace):
            return InvariantViolation(
                name=self.name,
                message=f"trace shorter than required step {self.step}",
            )
        entry: TraceEntry = state.trace[self.step]
        observed = entry.post_state["registers"].get(self.register, 0)
        if observed != self.expected:
            return InvariantViolation(
                name=self.name,
                message=(
                    f"expected r{self.register}={self.expected!r} at step {self.step}, observed {observed!r}"
                ),
                step=self.step,
            )
        return None


def evaluate_invariants(state: VMState, invariants: Sequence[Invariant]) -> tuple[InvariantViolation, ...]:
    violations: list[InvariantViolation] = []
    for invariant in invariants:
        violation = invariant.check(state)
        if violation is not None:
            violations.append(violation)
    return tuple(violations)
