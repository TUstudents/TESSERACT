from typing import Protocol, Any


class Critic(Protocol):
    def analyze(self, trace: Any, expected: Any | None = None) -> dict:
        ...
