"""Trace critic and repair loop scaffolding."""

from .differential import DifferentialCritic, ProgramExecution
from .interface import Critic
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
]
