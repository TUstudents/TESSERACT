from typing import Protocol, Sequence

from tesseract.vm.ir import Instruction


class Compiler(Protocol):
    def compile(self, prompt: str) -> Sequence[Instruction]:
        ...
