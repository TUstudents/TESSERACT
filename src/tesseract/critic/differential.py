from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from .interface import Critic, CriticInput
from .invariants import Invariant, evaluate_invariants
from .repair import build_repair_prompt
from .schema import CriticReport, FailureType, coerce_trace_entries, summarize_trace
from tesseract.vm import Instruction, Trap, VM, VMState

CONTROL_FLOW_OPCODES = {"JMP", "JZ", "JNZ", "JLT", "JGT", "CALL", "RET"}
TRAP_TO_FAILURE_TYPE: dict[str, FailureType] = {
    "TYPE": "TYPE_ERROR",
    "TIMEOUT": "TIMEOUT",
    "INVALID_OP": "INVALID_OP",
    "ADDR": "WRONG_ADDRESS",
    "DIV0": "WRONG_VALUE",
    "OVERFLOW": "WRONG_VALUE",
}


@dataclass(frozen=True)
class ProgramExecution:
    state: VMState
    trap: Trap | None = None


class DifferentialCritic(Critic):
    def analyze(self, trace: CriticInput, expected: CriticInput | None = None) -> dict:
        if expected is None:
            raise ValueError("expected trace or state is required for differential analysis")
        report = self.compare(trace, expected)
        return report.to_dict()

    def compare(
        self,
        candidate: CriticInput,
        expected: CriticInput,
        *,
        invariants: Sequence[Invariant] = (),
        task_prompt: str | None = None,
    ) -> CriticReport:
        candidate_state = self._coerce_state(candidate)
        expected_state = self._coerce_state(expected)

        candidate_summary = summarize_trace(candidate_state)
        expected_summary = summarize_trace(expected_state)
        invariant_violations = evaluate_invariants(candidate_state, invariants)
        first_failing_step = self._first_failing_step(candidate_state, expected_state)
        failure_type = self._classify_failure(candidate_state, expected_state, first_failing_step)
        if failure_type == "SUCCESS" and invariant_violations:
            failure_type = "INVARIANT_VIOLATION"
        differing_registers = self._differing_registers(candidate_state, expected_state, first_failing_step)
        differing_addresses = self._differing_addresses(candidate_state, expected_state, first_failing_step)
        message = self._build_message(failure_type, first_failing_step, candidate_summary, expected_summary)

        report = CriticReport(
            status="success" if failure_type == "SUCCESS" else "failure",
            failure_type=failure_type,
            first_failing_step=first_failing_step,
            message=message,
            candidate_summary=candidate_summary,
            expected_summary=expected_summary,
            differing_registers=differing_registers,
            differing_addresses=differing_addresses,
            invariant_violations=invariant_violations,
            metadata={
                "candidate_trace_length": len(candidate_state.trace),
                "expected_trace_length": len(expected_state.trace),
            },
        )
        if task_prompt is not None:
            object.__setattr__(report, "repair_prompt", build_repair_prompt(task_prompt, report))
        return report

    def compare_programs(
        self,
        vm: VM,
        candidate_program: Sequence[Instruction],
        expected_program: Sequence[Instruction],
        *,
        state: VMState | None = None,
        invariants: Sequence[Invariant] = (),
        task_prompt: str | None = None,
    ) -> CriticReport:
        candidate_execution = self._execute(vm, candidate_program, state)
        expected_execution = self._execute(vm, expected_program, state)
        return self.compare(
            candidate_execution.state,
            expected_execution.state,
            invariants=invariants,
            task_prompt=task_prompt,
        )

    def _execute(self, vm: VM, program: Sequence[Instruction], state: VMState | None) -> ProgramExecution:
        exec_state = VMState() if state is None else VMState(
            registers=dict(state.registers),
            memory=dict(state.memory),
            stack=list(state.stack),
            call_stack=list(state.call_stack),
            pc=state.pc,
            flags=dict(state.flags),
            halted=state.halted,
            halt_reason=state.halt_reason,
            step_count=state.step_count,
            trace=list(state.trace),
        )
        try:
            final_state = vm.execute(program, state=exec_state, trace=True)
            return ProgramExecution(state=final_state)
        except Trap as trap:
            return ProgramExecution(state=exec_state, trap=trap)

    def _coerce_state(self, trace_or_state: CriticInput) -> VMState:
        if isinstance(trace_or_state, VMState):
            return trace_or_state
        entries = coerce_trace_entries(trace_or_state)
        state = VMState(trace=list(entries))
        if entries:
            last = entries[-1]
            state.registers = dict(last.post_state["registers"])
            state.memory = dict(last.post_state["memory"])
            state.stack = list(last.post_state["stack"])
            state.call_stack = list(last.post_state["call_stack"])
            state.pc = last.post_state["pc"]
            state.flags = dict(last.post_state["flags"])
            state.halted = bool(last.post_state["halted"])
            state.halt_reason = last.post_state["halt_reason"]
            state.step_count = int(last.post_state["step_count"])
        return state

    def _first_failing_step(self, candidate: VMState, expected: VMState) -> int | None:
        limit = min(len(candidate.trace), len(expected.trace))
        for index in range(limit):
            candidate_entry = candidate.trace[index]
            expected_entry = expected.trace[index]
            if (
                candidate_entry.pc != expected_entry.pc
                or candidate_entry.instruction != expected_entry.instruction
                or candidate_entry.post_state != expected_entry.post_state
                or candidate_entry.trap != expected_entry.trap
            ):
                return index
        if len(candidate.trace) != len(expected.trace):
            return limit
        return None

    def _classify_failure(
        self,
        candidate: VMState,
        expected: VMState,
        first_failing_step: int | None,
    ) -> FailureType:
        if first_failing_step is None:
            if self._states_equivalent(candidate, expected):
                return "SUCCESS"
            candidate_trap = candidate.trace[-1].trap if candidate.trace and candidate.trace[-1].trap is not None else None
            if candidate_trap is not None:
                return TRAP_TO_FAILURE_TYPE.get(candidate_trap, "UNKNOWN_FAILURE")
            if candidate.halt_reason in TRAP_TO_FAILURE_TYPE:
                return TRAP_TO_FAILURE_TYPE[candidate.halt_reason]
            differing_addresses = self._memory_differences(candidate.memory, expected.memory)
            if differing_addresses:
                return "WRONG_ADDRESS"
            differing_registers = self._register_differences(candidate.registers, expected.registers)
            if differing_registers:
                return "WRONG_REGISTER"
            return "WRONG_VALUE"

        candidate_trap = candidate.trace[-1].trap if candidate.trace and candidate.trace[-1].trap is not None else None
        if candidate_trap is not None:
            return TRAP_TO_FAILURE_TYPE.get(candidate_trap, "UNKNOWN_FAILURE")
        if candidate.halt_reason in TRAP_TO_FAILURE_TYPE:
            return TRAP_TO_FAILURE_TYPE[candidate.halt_reason]
        if first_failing_step >= len(candidate.trace) or first_failing_step >= len(expected.trace):
            return "UNKNOWN_FAILURE"

        candidate_entry = candidate.trace[first_failing_step]
        expected_entry = expected.trace[first_failing_step]
        if (
            candidate_entry.pc != expected_entry.pc
            or candidate_entry.instruction.opcode != expected_entry.instruction.opcode
        ):
            if candidate_entry.instruction.opcode in CONTROL_FLOW_OPCODES or expected_entry.instruction.opcode in CONTROL_FLOW_OPCODES:
                return "WRONG_BRANCH"
        candidate_registers = candidate_entry.post_state["registers"]
        expected_registers = expected_entry.post_state["registers"]
        candidate_memory = candidate_entry.post_state["memory"]
        expected_memory = expected_entry.post_state["memory"]

        differing_addresses = self._memory_differences(candidate_memory, expected_memory)
        if differing_addresses:
            return "WRONG_ADDRESS"
        differing_registers = self._register_differences(candidate_registers, expected_registers)
        if differing_registers:
            return "WRONG_REGISTER"
        return "WRONG_VALUE"

    def _differing_registers(
        self,
        candidate: VMState,
        expected: VMState,
        first_failing_step: int | None,
    ) -> tuple[int, ...]:
        if first_failing_step is None:
            if self._states_equivalent(candidate, expected):
                return ()
            return self._register_differences(candidate.registers, expected.registers)
        if first_failing_step >= len(candidate.trace) or first_failing_step >= len(expected.trace):
            return ()
        candidate_registers = candidate.trace[first_failing_step].post_state["registers"]
        expected_registers = expected.trace[first_failing_step].post_state["registers"]
        return self._register_differences(candidate_registers, expected_registers)

    def _differing_addresses(
        self,
        candidate: VMState,
        expected: VMState,
        first_failing_step: int | None,
    ) -> tuple[int, ...]:
        if first_failing_step is None:
            if self._states_equivalent(candidate, expected):
                return ()
            return self._memory_differences(candidate.memory, expected.memory)
        if first_failing_step >= len(candidate.trace) or first_failing_step >= len(expected.trace):
            return ()
        candidate_memory = candidate.trace[first_failing_step].post_state["memory"]
        expected_memory = expected.trace[first_failing_step].post_state["memory"]
        return self._memory_differences(candidate_memory, expected_memory)

    def _states_equivalent(self, candidate: VMState, expected: VMState) -> bool:
        return (
            candidate.registers == expected.registers
            and candidate.memory == expected.memory
            and candidate.stack == expected.stack
            and candidate.call_stack == expected.call_stack
            and candidate.pc == expected.pc
            and candidate.flags == expected.flags
            and candidate.halted == expected.halted
            and candidate.halt_reason == expected.halt_reason
            and candidate.step_count == expected.step_count
        )

    def _register_differences(
        self,
        candidate_registers: Mapping[int, object],
        expected_registers: Mapping[int, object],
    ) -> tuple[int, ...]:
        registers = sorted(
            register
            for register in set(candidate_registers) | set(expected_registers)
            if candidate_registers.get(register) != expected_registers.get(register)
        )
        return tuple(registers)

    def _memory_differences(
        self,
        candidate_memory: Mapping[int, object],
        expected_memory: Mapping[int, object],
    ) -> tuple[int, ...]:
        addresses = sorted(
            address
            for address in set(candidate_memory) | set(expected_memory)
            if candidate_memory.get(address) != expected_memory.get(address)
        )
        return tuple(addresses)

    def _build_message(
        self,
        failure_type: FailureType,
        first_failing_step: int | None,
        candidate_summary: object,
        expected_summary: object,
    ) -> str:
        if failure_type == "SUCCESS":
            return "candidate trace matches expected trace"
        if failure_type == "INVARIANT_VIOLATION":
            return "candidate trace matches expected trace but violates one or more invariants"
        if first_failing_step is None:
            return f"execution failed with {failure_type}"
        return (
            f"first failing step {first_failing_step}: {failure_type}; "
            f"candidate={candidate_summary}; expected={expected_summary}"
        )
