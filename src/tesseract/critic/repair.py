from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, cast

from .schema import CriticReport, FailureType

_FAILURE_TYPES: tuple[FailureType, ...] = (
    "SUCCESS",
    "WRONG_BRANCH",
    "WRONG_REGISTER",
    "WRONG_ADDRESS",
    "WRONG_VALUE",
    "TYPE_ERROR",
    "TIMEOUT",
    "INVALID_OP",
    "INVARIANT_VIOLATION",
    "UNKNOWN_FAILURE",
)
_HALT_REASONS: tuple[str, ...] = ("HALT", "TIMEOUT", "INVALID_OP", "ADDR", "DIV0", "TYPE", "OVERFLOW")


@dataclass(frozen=True)
class RepairState:
    failure_type: FailureType
    first_failing_step: int | None
    candidate_halt_reason: str | None
    expected_halt_reason: str | None
    differing_registers: tuple[int, ...] = ()
    differing_addresses: tuple[int, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RepairState:
        return cls(
            failure_type=cast(FailureType, data["failure_type"]),
            first_failing_step=cast(int | None, data.get("first_failing_step")),
            candidate_halt_reason=cast(str | None, data.get("candidate_halt_reason")),
            expected_halt_reason=cast(str | None, data.get("expected_halt_reason")),
            differing_registers=tuple(int(value) for value in cast(list[int], data.get("differing_registers", []))),
            differing_addresses=tuple(int(value) for value in cast(list[int], data.get("differing_addresses", []))),
        )

    def to_text(self) -> str:
        first_step = "unknown" if self.first_failing_step is None else str(self.first_failing_step)
        return " | ".join(
            [
                f"failure={self.failure_type}",
                f"first_step={first_step}",
                f"candidate_halt={self.candidate_halt_reason}",
                f"expected_halt={self.expected_halt_reason}",
                f"registers={list(self.differing_registers)}",
                f"addresses={list(self.differing_addresses)}",
            ]
        )

    def feature_vector(self, *, max_registers: int = 4, max_addresses: int = 4) -> tuple[float, ...]:
        features: list[float] = [1.0 if self.failure_type == failure_type else 0.0 for failure_type in _FAILURE_TYPES]
        features.append(float(self.first_failing_step) if self.first_failing_step is not None else -1.0)
        features.extend(self._halt_reason_features(self.candidate_halt_reason))
        features.extend(self._halt_reason_features(self.expected_halt_reason))
        features.append(float(len(self.differing_registers)))
        features.append(float(len(self.differing_addresses)))
        features.extend(self._index_features(self.differing_registers, max_registers))
        features.extend(self._index_features(self.differing_addresses, max_addresses))
        return tuple(features)

    def _halt_reason_features(self, halt_reason: str | None) -> list[float]:
        return [1.0 if halt_reason == known else 0.0 for known in _HALT_REASONS] + [1.0 if halt_reason not in _HALT_REASONS else 0.0]

    def _index_features(self, values: tuple[int, ...], limit: int) -> list[float]:
        padded = list(values[:limit])
        padded.extend([-1] * (limit - len(padded)))
        return [float(value) for value in padded]


def repair_state_feature_dim() -> int:
    return len(_FAILURE_TYPES) + 1 + (2 * (len(_HALT_REASONS) + 1)) + 2 + 4 + 4


def build_repair_state(report: CriticReport) -> RepairState:
    expected_halt_reason = report.expected_summary.halt_reason if report.expected_summary is not None else None
    return RepairState(
        failure_type=report.failure_type,
        first_failing_step=report.first_failing_step,
        candidate_halt_reason=report.candidate_summary.halt_reason,
        expected_halt_reason=expected_halt_reason,
        differing_registers=report.differing_registers,
        differing_addresses=report.differing_addresses,
    )


def build_repair_prompt(task_prompt: str, report: CriticReport) -> str:
    repair_state = build_repair_state(report)
    pieces = [
        f"task: {task_prompt}",
        repair_state.to_text(),
        f"message: {report.message}",
    ]
    if report.invariant_violations:
        pieces.append(
            "invariants: "
            + "; ".join(f"{violation.name}: {violation.message}" for violation in report.invariant_violations)
        )
    return "\n".join(pieces)
