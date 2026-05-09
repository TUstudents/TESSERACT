from __future__ import annotations

import json

from .benchmark import BenchmarkReport, BenchmarkSuite, benchmark_suite_from_dict
from .research import ExperimentManifest, ResearchEvaluationReport


def benchmark_suite_to_json(suite: BenchmarkSuite) -> str:
    return json.dumps(suite.to_dict(), sort_keys=True)


def benchmark_suite_from_json(payload: str) -> BenchmarkSuite:
    return benchmark_suite_from_dict(json.loads(payload))


def benchmark_report_to_json(report: BenchmarkReport) -> str:
    return json.dumps(report.to_dict(), sort_keys=True)


def benchmark_report_to_text(report: BenchmarkReport) -> str:
    lines = [
        f"suite: {report.suite_name}",
        f"seed: {report.seed}",
        f"exact_output_accuracy: {report.exact_output_accuracy:.3f}",
        f"compile_validity_rate: {report.compile_validity_rate:.3f}",
        f"execution_success_rate: {report.execution_success_rate:.3f}",
        f"exact_program_match: {report.exact_program_match:.3f}",
        f"average_program_length: {report.average_program_length:.3f}",
        f"shortcut_rate: {report.shortcut_rate:.3f}",
        f"macro_step_efficiency: {report.macro_step_efficiency:.3f}",
    ]
    compile_breakdown = report.compile_failure_breakdown()
    if compile_breakdown:
        lines.append(
            "compile_failures: " + ", ".join(f"{label}={value:.3f}" for label, value in compile_breakdown.items())
        )
    execution_breakdown = report.execution_failure_breakdown()
    if execution_breakdown:
        lines.append(
            "execution_failures: " + ", ".join(f"{label}={value:.3f}" for label, value in execution_breakdown.items())
        )
    trace_summary = report.trace_length_summary()
    lines.append(
        "trace_lengths: "
        f"avg={trace_summary['average_trace_length']:.3f} "
        f"gold_avg={trace_summary['average_gold_trace_length']:.3f} "
        f"max={trace_summary['max_trace_length']:.3f}"
    )
    performance = report.performance_summary()
    lines.append(
        "performance_ms: "
        f"compile={performance['average_compile_time_ms']:.3f} "
        f"execute={performance['average_execute_time_ms']:.3f}"
    )
    for task_type, metrics in report.task_type_metrics().items():
        lines.append(
            f"task_type[{task_type}]: output={metrics['exact_output_accuracy']:.3f} "
            f"exec={metrics['execution_success_rate']:.3f} "
            f"program={metrics['exact_program_match']:.3f} "
            f"macro={metrics['macro_step_efficiency']:.3f}"
        )
    return "\n".join(lines)


def experiment_manifest_to_json(manifest: ExperimentManifest) -> str:
    return json.dumps(manifest.to_dict(), sort_keys=True)


def experiment_manifest_from_json(payload: str) -> ExperimentManifest:
    return ExperimentManifest.from_dict(json.loads(payload))


def research_evaluation_report_to_json(report: ResearchEvaluationReport) -> str:
    return json.dumps(report.to_dict(), sort_keys=True)


def research_evaluation_report_to_text(report: ResearchEvaluationReport) -> str:
    lines = [
        f"experiment: {report.manifest.experiment_name}",
        f"seed: {report.manifest.seed}",
        f"suite_payload_bytes: {len(report.manifest.suite_payload)}",
        "[exact_execution]",
        benchmark_report_to_text(report.exact_execution),
    ]
    if report.critic_localization is not None:
        lines.extend(
            [
                "[critic_localization]",
                f"failure_type_accuracy: {report.critic_localization.failure_type_accuracy:.3f}",
                f"first_step_accuracy: {report.critic_localization.first_step_accuracy:.3f}",
            ]
        )
    if report.repair_improvement is not None:
        lines.extend(
            [
                "[repair_improvement]",
                f"baseline_success_rate: {report.repair_improvement.baseline_success_rate:.3f}",
                f"repaired_success_rate: {report.repair_improvement.repaired_success_rate:.3f}",
                f"average_improvement: {report.repair_improvement.average_improvement:.3f}",
                f"success_after_1_round: {report.repair_improvement.metrics.success_after_1_round:.3f}",
                f"success_after_2_rounds: {report.repair_improvement.metrics.success_after_2_rounds:.3f}",
                f"success_after_3_rounds: {report.repair_improvement.metrics.success_after_3_rounds:.3f}",
            ]
        )
    if report.anti_shortcut is not None:
        lines.extend(
            [
                "[anti_shortcut]",
                f"faithful_execution_rate: {report.anti_shortcut.faithful_execution_rate:.3f}",
                f"corrupted_program_accuracy: {report.anti_shortcut.corrupted_program_accuracy:.3f}",
                f"degradation: {report.anti_shortcut.degradation:.3f}",
            ]
        )
    return "\n".join(lines)
