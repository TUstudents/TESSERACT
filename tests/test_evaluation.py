from __future__ import annotations

import json
from types import MethodType
from typing import Any, cast

import numpy as np
import torch

from tesseract.backbone import RuleBasedBackbone, generate_nl_tasks
from tesseract.compiler import build_training_batch, build_vocabularies, train_step
from tesseract.compiler.nl import BackboneConditionedCompiler
from tesseract.vm import Instruction, VM
from tesseract.evaluation import (
    benchmark_report_to_json,
    benchmark_report_to_text,
    benchmark_suite_from_json,
    benchmark_suite_to_json,
    build_nl_benchmark_suite,
    run_nl_benchmark,
    set_global_seed,
)


def _build_trained_nl_compiler() -> BackboneConditionedCompiler:
    nl_tasks = generate_nl_tasks(task_types=("arithmetic", "max", "sum_to_n"), operations=("add", "sub"), values=(1, 2, 3), seed=0)
    synthetic_tasks = [task.to_synthetic_task() for task in nl_tasks]
    artifacts = build_vocabularies(synthetic_tasks)
    batch = build_training_batch(
        synthetic_tasks,
        prompt_vocab=artifacts.prompt_vocab,
        program_tokenizer=artifacts.program_tokenizer,
    )
    train_step(artifacts.compiler.model, batch)
    return BackboneConditionedCompiler(backbone=RuleBasedBackbone(), compiler=artifacts.compiler)


def test_set_global_seed_is_reproducible() -> None:
    set_global_seed(123)
    first_torch = torch.rand(4)
    first_numpy = np.random.rand(4)
    set_global_seed(123)
    second_torch = torch.rand(4)
    second_numpy = np.random.rand(4)

    assert torch.equal(first_torch, second_torch)
    assert np.array_equal(first_numpy, second_numpy)


def test_nl_benchmark_suite_is_seed_reproducible() -> None:
    first = build_nl_benchmark_suite(seed=7)
    second = build_nl_benchmark_suite(seed=7)
    third = build_nl_benchmark_suite(seed=8)

    assert first.tasks == second.tasks
    assert first.tasks != third.tasks


def test_benchmark_suite_round_trips_through_json_freeze_payload() -> None:
    suite = build_nl_benchmark_suite(seed=7)

    payload = benchmark_suite_to_json(suite)
    restored = benchmark_suite_from_json(payload)

    assert restored == suite


def test_run_nl_benchmark_reports_exact_execution_metrics() -> None:
    compiler = _build_trained_nl_compiler()
    suite = build_nl_benchmark_suite(seed=0)

    report = run_nl_benchmark(compiler, suite)

    assert report.exact_output_accuracy == 1.0
    assert report.compile_validity_rate == 1.0
    assert report.execution_success_rate == 1.0
    assert report.exact_program_match == 1.0
    assert report.average_program_length > 4.0


def test_run_nl_benchmark_records_timeout_failures_stably() -> None:
    compiler = _build_trained_nl_compiler()
    suite = build_nl_benchmark_suite(seed=0)

    def looping_compile(self: Any, prompt: str):
        del prompt
        return (Instruction("JMP", imm=0), Instruction("HALT"))

    cast(Any, compiler.compiler).compile = MethodType(looping_compile, compiler.compiler)
    report = run_nl_benchmark(compiler, suite, vm=VM(step_budget=5))

    assert report.compile_validity_rate == 1.0
    assert report.execution_success_rate == 0.0
    assert {result.trap_kind for result in report.results} == {"TIMEOUT"}


def test_benchmark_report_serialization_helpers() -> None:
    compiler = _build_trained_nl_compiler()
    suite = build_nl_benchmark_suite(seed=0)
    report = run_nl_benchmark(compiler, suite)

    payload = benchmark_report_to_json(report)
    text = benchmark_report_to_text(report)

    parsed = json.loads(payload)
    assert parsed["suite_name"] == suite.name
    assert "exact_output_accuracy" in parsed
    assert f"suite: {suite.name}" in text
    assert "execution_success_rate:" in text
