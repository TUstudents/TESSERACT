import importlib
import sys
from types import ModuleType

import pytest
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
    from tesseract.backbone import LearnedBackbone, RuleBasedBackbone, generate_nl_tasks
    from tesseract.compiler import BackboneConditionedCompiler
    from tesseract.evaluation import build_nl_benchmark_suite

    assert isinstance(RuleBasedBackbone(), RuleBasedBackbone)
    assert LearnedBackbone.__name__ == "LearnedBackbone"
    assert BackboneConditionedCompiler.__name__ == "BackboneConditionedCompiler"
    assert generate_nl_tasks(task_types=("arithmetic",), operations=("add",), values=(1,))
    assert build_nl_benchmark_suite(seed=0).tasks


def test_package_optional_imports_do_not_mask_compiler(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import_module = importlib.import_module
    cached_evaluation = sys.modules.pop("tesseract.evaluation", None)

    def guarded_import_module(name: str, package: str | None = None) -> ModuleType:
        if name == ".evaluation" and package == "tesseract":
            raise ModuleNotFoundError("synthetic missing torch", name="torch")
        return real_import_module(name, package)

    try:
        monkeypatch.setattr(importlib, "import_module", guarded_import_module)
        reloaded = importlib.reload(tesseract)

        assert reloaded.compiler is not None
        assert reloaded.evaluation is None
    finally:
        monkeypatch.setattr(importlib, "import_module", real_import_module)
        if cached_evaluation is not None:
            sys.modules["tesseract.evaluation"] = cached_evaluation
        restored_evaluation = importlib.import_module("tesseract.evaluation")
        restored = importlib.reload(tesseract)
        assert restored.compiler is not None
        assert restored_evaluation is not None


def test_package_optional_imports_do_not_mask_internal_import_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import_module = importlib.import_module

    def guarded_import_module(name: str, package: str | None = None) -> ModuleType:
        if name == ".evaluation" and package == "tesseract":
            raise ModuleNotFoundError("synthetic internal failure", name="tesseract.evaluation.internal")
        return real_import_module(name, package)

    try:
        monkeypatch.setattr(importlib, "import_module", guarded_import_module)
        with pytest.raises(ModuleNotFoundError, match="synthetic internal failure"):
            importlib.reload(tesseract)
    finally:
        monkeypatch.setattr(importlib, "import_module", real_import_module)
        importlib.reload(tesseract)
