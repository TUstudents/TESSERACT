"""Trace critic and repair loop scaffolding."""

from .differential import DifferentialCritic, ProgramExecution
from .interface import Critic, CriticInput
from .loop import RepairAttempt, RepairContext, RepairLoopController, RepairLoopMetrics, RepairLoopResult, evaluate_repair_loop
from .invariants import (
    FinalMemoryInvariant,
    FinalRegisterInvariant,
    Invariant,
    MaxStepsInvariant,
    NoTrapInvariant,
    TraceStepInvariant,
    evaluate_invariants,
)
from .repair import build_repair_prompt
from .schema import CriticReport, FailureType, InvariantViolation, TraceStatus, TraceSummary, summarize_trace

__all__ = [
    "Critic",
    "CriticInput",
    "TraceStatus",
    "FailureType",
    "TraceSummary",
    "InvariantViolation",
    "CriticReport",
    "summarize_trace",
    "Invariant",
    "NoTrapInvariant",
    "FinalRegisterInvariant",
    "FinalMemoryInvariant",
    "MaxStepsInvariant",
    "TraceStepInvariant",
    "evaluate_invariants",
    "build_repair_prompt",
    "ProgramExecution",
    "DifferentialCritic",
    "RepairContext",
    "RepairAttempt",
    "RepairLoopResult",
    "RepairLoopMetrics",
    "RepairLoopController",
    "evaluate_repair_loop",
]
