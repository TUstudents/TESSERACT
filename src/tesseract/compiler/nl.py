from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

from tesseract.backbone.interface import Backbone, BackboneOutput
from tesseract.compiler.baseline import AutoregressiveCompiler
from tesseract.critic.schema import CriticReport
from tesseract.vm import Instruction, VM, validate_program


@dataclass(frozen=True)
class NaturalLanguageCompileResult:
    backbone_output: BackboneOutput
    program: tuple[Instruction, ...]


@dataclass(frozen=True)
class NaturalLanguageExecutionResult:
    backbone_output: BackboneOutput
    program: tuple[Instruction, ...]
    output: int


class RepairCapableCompiler(Protocol):
    def compile_with_backbone_output(
        self,
        prompt: str,
        *,
        repair_context: CriticReport | None = None,
    ) -> NaturalLanguageCompileResult:
        ...

    def repair_compile(
        self,
        prompt: str,
        report: CriticReport,
    ) -> NaturalLanguageCompileResult:
        ...


@dataclass
class BackboneConditionedCompiler:
    backbone: Backbone
    compiler: AutoregressiveCompiler

    def compile(self, prompt: str) -> Sequence[Instruction]:
        return self.compile_with_backbone_output(prompt).program

    def compile_with_backbone_output(
        self,
        prompt: str,
        *,
        repair_context: CriticReport | None = None,
    ) -> NaturalLanguageCompileResult:
        repair_hint = repair_context.repair_prompt if repair_context is not None else None
        backbone_output = self.backbone.encode(prompt, repair_hint=repair_hint)
        conditioning = backbone_output.conditioning or None
        program = tuple(self.compiler.compile_conditioned(backbone_output.canonical_prompt, conditioning))
        return NaturalLanguageCompileResult(backbone_output=backbone_output, program=program)

    def repair_compile(
        self,
        prompt: str,
        report: CriticReport,
    ) -> NaturalLanguageCompileResult:
        return self.compile_with_backbone_output(prompt, repair_context=report)

    def execute(
        self,
        prompt: str,
        *,
        vm: VM | None = None,
        repair_context: CriticReport | None = None,
    ) -> NaturalLanguageExecutionResult:
        compile_result = self.compile_with_backbone_output(prompt, repair_context=repair_context)
        validate_program(compile_result.program)
        machine = vm if vm is not None else VM()
        state = machine.execute(compile_result.program)
        return NaturalLanguageExecutionResult(
            backbone_output=compile_result.backbone_output,
            program=compile_result.program,
            output=state.registers.get(compile_result.backbone_output.result_register, 0),
        )
