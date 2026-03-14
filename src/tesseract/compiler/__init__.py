"""Latent compiler interfaces, baselines, and NL-conditioned compiler paths."""

from importlib import import_module
from typing import TYPE_CHECKING, Any

from .baseline import (
    AutoregressiveCompiler,
    AutoregressiveCompilerModel,
    CountBasedAutoregressiveCompiler,
    CountBasedAutoregressiveCompilerModel,
    ProgramTokenizer,
    ProgramVocabulary,
    PromptVocabulary,
)
from .interface import Compiler
from .synthetic import (
    RESULT_REGISTER,
    SUPPORTED_OPERATIONS,
    SUPPORTED_TASK_TYPES,
    SyntheticTask,
    TaskExecutionResult,
    build_arithmetic_prompt,
    build_gold_program,
    build_max_program,
    build_max_prompt,
    build_sum_to_n_program,
    build_sum_to_n_prompt,
    evaluate_operation,
    execute_task,
    generate_synthetic_tasks,
    make_max_task,
    make_sum_to_n_task,
    make_synthetic_task,
)
from .training import (
    CompilerArtifacts,
    EvaluationMetrics,
    TrainingBatch,
    build_training_batch,
    build_vocabularies,
    evaluate_compiler,
    train_step,
)

if TYPE_CHECKING:
    from .nl import BackboneConditionedCompiler, NaturalLanguageCompileResult, NaturalLanguageExecutionResult, RepairCapableCompiler

__all__ = [
    "Compiler",
    "BackboneConditionedCompiler",
    "NaturalLanguageCompileResult",
    "NaturalLanguageExecutionResult",
    "RepairCapableCompiler",
    "RESULT_REGISTER",
    "SUPPORTED_OPERATIONS",
    "SUPPORTED_TASK_TYPES",
    "SyntheticTask",
    "TaskExecutionResult",
    "build_arithmetic_prompt",
    "build_gold_program",
    "build_max_program",
    "build_max_prompt",
    "build_sum_to_n_program",
    "build_sum_to_n_prompt",
    "evaluate_operation",
    "execute_task",
    "generate_synthetic_tasks",
    "make_max_task",
    "make_sum_to_n_task",
    "make_synthetic_task",
    "PromptVocabulary",
    "ProgramVocabulary",
    "ProgramTokenizer",
    "CountBasedAutoregressiveCompilerModel",
    "CountBasedAutoregressiveCompiler",
    "AutoregressiveCompilerModel",
    "AutoregressiveCompiler",
    "CompilerArtifacts",
    "EvaluationMetrics",
    "TrainingBatch",
    "build_vocabularies",
    "build_training_batch",
    "train_step",
    "evaluate_compiler",
]


def __getattr__(name: str) -> Any:
    if name in {
        "BackboneConditionedCompiler",
        "NaturalLanguageCompileResult",
        "NaturalLanguageExecutionResult",
        "RepairCapableCompiler",
    }:
        module = import_module(".nl", package=__name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
