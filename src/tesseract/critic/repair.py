from __future__ import annotations

from .schema import CriticReport


def build_repair_prompt(task_prompt: str, report: CriticReport) -> str:
    first_step = "unknown" if report.first_failing_step is None else str(report.first_failing_step)
    pieces = [
        f"task: {task_prompt}",
        f"failure_type: {report.failure_type}",
        f"first_failing_step: {first_step}",
        f"candidate_halt_reason: {report.candidate_summary.halt_reason}",
        f"message: {report.message}",
    ]
    if report.differing_registers:
        pieces.append(f"differing_registers: {list(report.differing_registers)}")
    if report.differing_addresses:
        pieces.append(f"differing_addresses: {list(report.differing_addresses)}")
    if report.invariant_violations:
        pieces.append(
            "invariants: "
            + "; ".join(f"{violation.name}: {violation.message}" for violation in report.invariant_violations)
        )
    return "\n".join(pieces)
