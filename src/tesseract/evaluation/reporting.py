from __future__ import annotations

import json

from .benchmark import BenchmarkReport, BenchmarkSuite, benchmark_suite_from_dict


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
    ]
    for task_type, metrics in report.task_type_metrics().items():
        lines.append(
            f"task_type[{task_type}]: output={metrics['exact_output_accuracy']:.3f} "
            f"exec={metrics['execution_success_rate']:.3f} "
            f"program={metrics['exact_program_match']:.3f}"
        )
    return "\n".join(lines)
