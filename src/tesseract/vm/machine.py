from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .ir import Instruction
from .state import TraceEntry, VMState, VMValue


@dataclass
class Trap(Exception):
    kind: str
    pc: int | None = None
    instruction: Instruction | None = None

    def __str__(self) -> str:
        if self.pc is None:
            return self.kind
        return f"{self.kind} at pc={self.pc}"


class VM:
    def __init__(
        self,
        step_budget: int = 10_000,
        register_count: int = 32,
        memory_size: int = 65_536,
    ):
        self.step_budget = step_budget
        self.register_count = register_count
        self.memory_size = memory_size

    def execute(
        self,
        program: Iterable[Instruction],
        state: VMState | None = None,
        *,
        trace: bool = False,
    ) -> VMState:
        prog = list(program)
        s = state if state is not None else VMState()
        if trace:
            s.trace = []

        while 0 <= s.pc < len(prog) and not s.halted:
            if s.step_count >= self.step_budget:
                trap = Trap("TIMEOUT", pc=s.pc, instruction=prog[s.pc])
                s.halted = True
                s.halt_reason = trap.kind
                raise trap

            current_pc = s.pc
            ins = prog[current_pc]
            pre_state = s.snapshot() if trace else None

            try:
                self._execute_instruction(s, ins, len(prog))
            except Trap as trap:
                s.halted = True
                s.halt_reason = trap.kind
                if trace and pre_state is not None:
                    s.trace.append(
                        TraceEntry(
                            step=s.step_count,
                            pc=current_pc,
                            instruction=ins,
                            pre_state=pre_state,
                            post_state=s.snapshot(),
                            trap=trap.kind,
                        )
                    )
                raise

            s.step_count += 1
            if trace and pre_state is not None:
                s.trace.append(
                    TraceEntry(
                        step=s.step_count - 1,
                        pc=current_pc,
                        instruction=ins,
                        pre_state=pre_state,
                        post_state=s.snapshot(),
                    )
                )

        return s

    def _execute_instruction(self, state: VMState, ins: Instruction, program_length: int) -> None:
        opcode = ins.opcode
        if opcode == "HALT":
            state.halted = True
            state.halt_reason = "HALT"
            state.pc += 1
        elif opcode == "CONST":
            dst = self._require_register_index(ins.dst, ins)
            if ins.imm is None:
                raise Trap("INVALID_OP", pc=state.pc, instruction=ins)
            self._write_register(state, dst, self._coerce_immediate(ins.imm, ins.type_tag), ins)
            state.pc += 1
        elif opcode == "MOV":
            dst = self._require_register_index(ins.dst, ins)
            src = self._require_register_index(ins.src1, ins)
            value = self._read_register(state, src, ins)
            self._write_register(state, dst, self._apply_type_tag(value, ins.type_tag), ins)
            state.pc += 1
        elif opcode in {"ADD", "SUB", "MUL", "DIV"}:
            dst = self._require_register_index(ins.dst, ins)
            lhs = self._read_int_register(state, ins.src1, ins)
            rhs = self._read_int_register(state, ins.src2, ins)
            if opcode == "ADD":
                result = lhs + rhs
            elif opcode == "SUB":
                result = lhs - rhs
            elif opcode == "MUL":
                result = lhs * rhs
            else:
                if rhs == 0:
                    raise Trap("DIV0", pc=state.pc, instruction=ins)
                result = int(lhs / rhs)
            result = self._apply_integer_tag(result, ins.type_tag, ins, state.pc)
            self._write_register(state, dst, result, ins)
            self._set_scalar_flags(state, result)
            state.pc += 1
        elif opcode in {"AND", "OR", "XOR"}:
            dst = self._require_register_index(ins.dst, ins)
            lhs = self._read_bool_register(state, ins.src1, ins)
            rhs = self._read_bool_register(state, ins.src2, ins)
            if opcode == "AND":
                result = lhs and rhs
            elif opcode == "OR":
                result = lhs or rhs
            else:
                result = lhs ^ rhs
            self._write_register(state, dst, result, ins)
            self._set_scalar_flags(state, result)
            state.pc += 1
        elif opcode == "NOT":
            dst = self._require_register_index(ins.dst, ins)
            value = self._read_bool_register(state, ins.src1, ins)
            result = not value
            self._write_register(state, dst, result, ins)
            self._set_scalar_flags(state, result)
            state.pc += 1
        elif opcode in {"CMP_EQ", "CMP_LT", "CMP_GT"}:
            dst = self._require_register_index(ins.dst, ins)
            lhs = self._read_register(state, self._require_register_index(ins.src1, ins), ins)
            rhs = self._read_register(state, self._require_register_index(ins.src2, ins), ins)
            if opcode == "CMP_EQ":
                result = type(lhs) is type(rhs) and lhs == rhs
                state.flags["eq"] = result
                state.flags["lt"] = False
                state.flags["gt"] = False
            else:
                if type(lhs) is not int or type(rhs) is not int:
                    raise Trap("TYPE", pc=state.pc, instruction=ins)
                if opcode == "CMP_LT":
                    result = lhs < rhs
                    state.flags["lt"] = result
                    state.flags["gt"] = False
                    state.flags["eq"] = lhs == rhs
                else:
                    result = lhs > rhs
                    state.flags["gt"] = result
                    state.flags["lt"] = False
                    state.flags["eq"] = lhs == rhs
            state.flags["zero"] = not result
            self._write_register(state, dst, result, ins)
            state.pc += 1
        elif opcode == "JMP":
            state.pc = self._branch_target(ins, state.pc, program_length)
        elif opcode in {"JZ", "JNZ"}:
            value = self._read_register(
                state,
                self._require_register_index(ins.src1, ins),
                ins,
            )
            is_zero = self._is_zero_value(value)
            should_jump = is_zero if opcode == "JZ" else not is_zero
            if should_jump:
                state.pc = self._branch_target(ins, state.pc, program_length)
            else:
                state.pc += 1
        elif opcode == "JLT":
            if state.flags["lt"]:
                state.pc = self._branch_target(ins, state.pc, program_length)
            else:
                state.pc += 1
        elif opcode == "JGT":
            if state.flags["gt"]:
                state.pc = self._branch_target(ins, state.pc, program_length)
            else:
                state.pc += 1
        elif opcode == "LOAD":
            dst = self._require_register_index(ins.dst, ins)
            address = self._memory_address(state, ins.src1, ins.imm, ins)
            value = state.memory.get(address, 0)
            self._write_register(state, dst, self._apply_type_tag(value, ins.type_tag), ins)
            state.pc += 1
        elif opcode == "STORE":
            value = self._read_register(
                state,
                self._require_register_index(ins.src2, ins),
                ins,
            )
            address = self._memory_address(state, ins.src1, ins.imm, ins)
            state.memory[address] = self._apply_type_tag(value, ins.type_tag)
            state.pc += 1
        elif opcode == "PUSH":
            value = self._read_register(
                state,
                self._require_register_index(ins.src1, ins),
                ins,
            )
            state.stack.append(value)
            state.pc += 1
        elif opcode == "POP":
            dst = self._require_register_index(ins.dst, ins)
            if not state.stack:
                raise Trap("ADDR", pc=state.pc, instruction=ins)
            value = state.stack.pop()
            self._write_register(state, dst, value, ins)
            state.pc += 1
        elif opcode == "CALL":
            state.call_stack.append(state.pc + 1)
            state.pc = self._branch_target(ins, state.pc, program_length)
        elif opcode == "RET":
            if not state.call_stack:
                raise Trap("ADDR", pc=state.pc, instruction=ins)
            state.pc = state.call_stack.pop()
        else:
            raise Trap("INVALID_OP", pc=state.pc, instruction=ins)

    def _branch_target(self, ins: Instruction, pc: int, program_length: int) -> int:
        if type(ins.imm) is not int:
            raise Trap("INVALID_OP", pc=pc, instruction=ins)
        if not 0 <= ins.imm < program_length:
            raise Trap("ADDR", pc=pc, instruction=ins)
        return ins.imm

    def _memory_address(
        self,
        state: VMState,
        base_register: int | None,
        offset: int | bool | None,
        ins: Instruction,
    ) -> int:
        base = self._read_int_register(state, base_register, ins)
        if offset is None:
            address = base
        else:
            if type(offset) is not int:
                raise Trap("TYPE", pc=state.pc, instruction=ins)
            address = base + offset
        if not 0 <= address < self.memory_size:
            raise Trap("ADDR", pc=state.pc, instruction=ins)
        return address

    def _require_register_index(self, index: int | None, ins: Instruction) -> int:
        if index is None or not 0 <= index < self.register_count:
            raise Trap("INVALID_OP", instruction=ins)
        return index

    def _read_register(self, state: VMState, index: int, ins: Instruction) -> VMValue:
        self._require_register_index(index, ins)
        return state.registers.get(index, 0)

    def _write_register(self, state: VMState, index: int, value: VMValue, ins: Instruction) -> None:
        self._require_register_index(index, ins)
        state.registers[index] = value

    def _read_int_register(self, state: VMState, index: int | None, ins: Instruction) -> int:
        value = self._read_register(state, self._require_register_index(index, ins), ins)
        if type(value) is not int:
            raise Trap("TYPE", pc=state.pc, instruction=ins)
        return value

    def _read_bool_register(self, state: VMState, index: int | None, ins: Instruction) -> bool:
        value = self._read_register(state, self._require_register_index(index, ins), ins)
        if type(value) is not bool:
            raise Trap("TYPE", pc=state.pc, instruction=ins)
        return value

    def _coerce_immediate(self, value: int | bool, type_tag: str | None) -> VMValue:
        if type_tag == "bool":
            if type(value) is bool:
                return value
            if type(value) is int and value in {0, 1}:
                return bool(value)
            raise Trap("TYPE")
        if type_tag in {None, "int", "i32", "i64", "checked_i32"}:
            if type(value) is bool:
                raise Trap("TYPE")
            if type(value) is not int:
                raise Trap("TYPE")
            return self._apply_integer_tag(value, type_tag, None, None)
        raise Trap("TYPE")

    def _apply_type_tag(self, value: VMValue, type_tag: str | None) -> VMValue:
        if type_tag is None:
            return value
        if type_tag == "bool":
            if type(value) is not bool:
                raise Trap("TYPE")
            return value
        if type(value) is not int:
            raise Trap("TYPE")
        return self._apply_integer_tag(value, type_tag, None, None)

    def _apply_integer_tag(
        self,
        value: int,
        type_tag: str | None,
        ins: Instruction | None,
        pc: int | None,
    ) -> int:
        if type_tag in {None, "int", "i64"}:
            return value
        if type_tag == "checked_i32":
            if not -(2**31) <= value < 2**31:
                raise Trap("OVERFLOW", pc=pc, instruction=ins)
            return value
        if type_tag == "i32":
            wrapped = value & 0xFFFF_FFFF
            if wrapped >= 2**31:
                wrapped -= 2**32
            return wrapped
        raise Trap("TYPE", pc=pc, instruction=ins)

    def _set_scalar_flags(self, state: VMState, value: VMValue) -> None:
        zero = self._is_zero_value(value)
        state.flags["zero"] = zero
        state.flags["eq"] = zero
        state.flags["lt"] = False
        state.flags["gt"] = False

    def _is_zero_value(self, value: VMValue) -> bool:
        if type(value) is bool:
            return value is False
        return value == 0
