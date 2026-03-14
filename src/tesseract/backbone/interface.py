from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass(frozen=True)
class BackboneOutput:
    original_prompt: str
    canonical_prompt: str
    task_type: str
    result_register: int
    values: tuple[int, ...] = ()
    conditioning: tuple[float, ...] = ()
    metadata: dict[str, int | float | str | bool] = field(default_factory=dict)


class Backbone(Protocol):
    def encode(
        self,
        prompt: str,
        *,
        repair_hint: str | None = None,
    ) -> BackboneOutput:
        ...
