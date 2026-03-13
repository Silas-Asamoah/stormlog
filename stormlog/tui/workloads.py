"""Sample workload helpers and output formatters for the Textual TUI."""

from __future__ import annotations

from typing import Any, cast

from stormlog.utils import format_bytes


def run_pytorch_sample_workload(profiler_cls: Any, torch_module: Any) -> dict[str, Any]:
    profiler = profiler_cls()

    def workload() -> Any:
        x = torch_module.randn((3072, 3072), device="cuda")
        y = torch_module.matmul(x, x)
        return y.sum()

    profiler.profile_function(workload)
    return cast(dict[str, Any], profiler.get_summary())


def run_tensorflow_sample_workload(profiler_cls: Any, tf_module: Any) -> Any:
    profiler = profiler_cls()
    with profiler.profile_context("tf_sample"):
        tensor = tf_module.random.normal((2048, 2048))
        product = tf_module.matmul(tensor, tensor)
        tf_module.reduce_sum(product)
    return profiler.get_results()


def run_cpu_sample_workload(profiler_cls: Any) -> dict[str, Any]:
    profiler = profiler_cls()

    def workload() -> int:
        data = [i for i in range(500000)]
        return sum(data)

    profiler.profile_function(workload)
    return cast(dict[str, Any], profiler.get_summary())


def format_pytorch_summary(summary: dict[str, Any]) -> str:
    peak = summary.get("peak_memory_usage", 0)
    delta = summary.get("memory_change_from_baseline", 0)
    delta_sign = "-" if delta < 0 else ""
    calls = summary.get("total_function_calls", "N/A")
    lines = [
        f"Functions profiled: {summary.get('total_functions_profiled', 'N/A')}",
        f"Total calls: {calls}",
        f"Peak memory: {format_bytes(peak)}",
        f"Δ from baseline: {delta_sign}{format_bytes(abs(delta))}",
    ]
    return "\n".join(lines)


def format_tensorflow_results(results: Any) -> str:
    duration = getattr(results, "duration", 0.0)
    peak_memory_mb = getattr(results, "peak_memory_mb", 0.0)
    average_memory_mb = getattr(results, "average_memory_mb", 0.0)
    snapshots = getattr(results, "snapshots", [])

    duration = 0.0 if duration is None else duration
    peak_memory_mb = 0.0 if peak_memory_mb is None else peak_memory_mb
    average_memory_mb = 0.0 if average_memory_mb is None else average_memory_mb
    snapshots = [] if snapshots is None else snapshots

    lines = [
        f"Duration: {duration:.2f}s",
        f"Peak memory: {peak_memory_mb:.2f} MB",
        f"Average memory: {average_memory_mb:.2f} MB",
        f"Snapshots: {len(snapshots)}",
    ]
    return "\n".join(lines)


def format_cpu_summary(summary: dict[str, Any]) -> str:
    delta = summary.get("memory_change_from_baseline", 0)
    delta_sign = "-" if delta < 0 else ""
    lines = [
        f"Snapshots collected: {summary.get('snapshots_collected', 0)}",
        f"Peak RSS: {format_bytes(summary.get('peak_memory_usage', 0))}",
        f"Δ from baseline: {delta_sign}{format_bytes(abs(delta))}",
    ]
    return "\n".join(lines)
