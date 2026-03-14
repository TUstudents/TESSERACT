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


def test_backbone_and_evaluation_submodule_imports() -> None:
    from tesseract.backbone import RuleBasedBackbone, generate_nl_tasks
    from tesseract.evaluation import build_nl_benchmark_suite

    assert isinstance(RuleBasedBackbone(), RuleBasedBackbone)
    assert generate_nl_tasks(task_types=("arithmetic",), operations=("add",), values=(1,))
    assert build_nl_benchmark_suite(seed=0).tasks
