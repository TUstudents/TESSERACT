import tesseract


def test_package_imports() -> None:
    assert tesseract.__all__ == ["backbone", "compiler", "vm", "critic"]


def test_vm_submodule_imports() -> None:
    from tesseract.vm.ir import Instruction
    from tesseract.vm.machine import VM
    from tesseract.vm.state import VMState

    assert Instruction(opcode="HALT").opcode == "HALT"
    assert isinstance(VM(), VM)
    assert isinstance(VMState(), VMState)
