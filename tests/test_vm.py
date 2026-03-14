from tesseract.vm.ir import Instruction
from tesseract.vm.machine import VM


def test_const_add_halt():
    vm = VM()
    program = [
        Instruction("CONST", dst=0, imm=2),
        Instruction("CONST", dst=1, imm=3),
        Instruction("ADD", dst=2, src1=0, src2=1),
        Instruction("HALT"),
    ]
    state = vm.execute(program)
    assert state.registers[2] == 5
