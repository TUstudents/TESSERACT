from typing import Protocol

from tesseract.vm.state import TraceEntry, VMState

CriticInput = VMState | list[TraceEntry] | tuple[TraceEntry, ...]


class Critic(Protocol):
    def analyze(self, trace: CriticInput, expected: CriticInput | None = None) -> dict:
        ...
