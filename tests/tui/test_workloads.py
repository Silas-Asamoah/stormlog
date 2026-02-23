from types import SimpleNamespace

from gpumemprof.tui.workloads import (
    format_cpu_summary,
    format_pytorch_summary,
    format_tensorflow_results,
)


def test_format_pytorch_summary_formats_negative_delta_in_gb() -> None:
    summary = {
        "total_functions_profiled": 1,
        "total_function_calls": 1,
        "peak_memory_usage": 0,
        "memory_change_from_baseline": -(1024**3),
    }

    formatted = format_pytorch_summary(summary)

    assert "from baseline: -1.00 GB" in formatted
    assert "-1073741824" not in formatted


def test_format_cpu_summary_formats_negative_delta_in_gb() -> None:
    summary = {
        "snapshots_collected": 1,
        "peak_memory_usage": 0,
        "memory_change_from_baseline": -(1024**3),
    }

    formatted = format_cpu_summary(summary)

    assert "from baseline: -1.00 GB" in formatted
    assert "-1073741824" not in formatted


def test_format_tensorflow_results_handles_missing_attributes() -> None:
    formatted = format_tensorflow_results(object())

    assert formatted == (
        "Duration: 0.00s\n"
        "Peak memory: 0.00 MB\n"
        "Average memory: 0.00 MB\n"
        "Snapshots: 0"
    )


def test_format_tensorflow_results_handles_none_values() -> None:
    results = SimpleNamespace(
        duration=None,
        peak_memory_mb=None,
        average_memory_mb=None,
        snapshots=None,
    )

    formatted = format_tensorflow_results(results)

    assert formatted == (
        "Duration: 0.00s\n"
        "Peak memory: 0.00 MB\n"
        "Average memory: 0.00 MB\n"
        "Snapshots: 0"
    )
