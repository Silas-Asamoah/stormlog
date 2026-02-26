"""Markdown and table-data builders for the Textual TUI."""

from __future__ import annotations

from textwrap import dedent
from typing import Any


def build_welcome_info() -> str:
    """Build welcome navigation guide text."""
    return dedent(
        """
        # Quick Start Guide

        ## Navigate the TUI

        Click on any tab above to explore different features:

        - **PyTorch** → View PyTorch GPU stats, run profiling samples, and see profile results
        - **TensorFlow** → View TensorFlow GPU stats, run profiling samples, and see profile results
        - **Monitoring** → Start live memory tracking, set alert thresholds, export CSV/JSON data
        - **Visualizations** → Generate timeline plots (PNG/HTML) from tracking sessions
        - **CLI & Actions** → Run CLI commands interactively and execute sample workloads

        ## Keyboard Shortcuts

        - **r** - Refresh overview tab
        - **g** - Log gpumemprof command examples
        - **t** - Log tfmemprof command examples
        - **f** - Focus log area in CLI tab
        - **q** - Quit application

        ## Getting Started

        1. **Check System Info** - Scroll down to see your platform, Python version, and GPU details
        2. **View GPU Stats** - Visit **PyTorch** or **TensorFlow** tabs to see real-time GPU memory statistics
        3. **Start Tracking** - Go to **Monitoring** tab and click "Start Live Tracking" to begin monitoring
        4. **Run Samples** - Use **CLI & Actions** tab to run sample workloads and see profiling results
        5. **Export Data** - After tracking, use "Export CSV" or "Export JSON" buttons in Monitoring tab

        ---
        """
    ).strip()


def build_system_markdown(
    *,
    system_info: dict[str, Any],
    gpu_info: dict[str, Any],
    tf_system_info: dict[str, Any],
    tf_gpu_info: dict[str, Any],
) -> str:
    lines = [
        "# System Overview",
        "",
        f"- **Platform**: {system_info.get('platform', 'Unknown')}",
        f"- **Python**: {system_info.get('python_version', 'Unknown')}",
        f"- **TensorFlow (Python)**: {tf_system_info.get('tensorflow_version', 'N/A')}",
        f"- **CUDA Available**: {system_info.get('cuda_available', False)}",
    ]

    if system_info.get("cuda_available"):
        lines.extend(
            [
                f"- **CUDA Version**: {system_info.get('cuda_version', 'Unknown')}",
                f"- **GPU Count**: {system_info.get('cuda_device_count', 0)}",
            ]
        )

    if gpu_info:
        lines.append("")
        lines.append("## GPU Snapshot")
        lines.extend(
            [
                f"- **Device Name**: {gpu_info.get('device_name', 'Unknown')}",
                f"- **Total Memory**: {gpu_info.get('total_memory', 0) / (1024**3):.2f} GB",
                f"- **Allocated**: {gpu_info.get('allocated_memory', 0) / (1024**3):.2f} GB",
                f"- **Reserved**: {gpu_info.get('reserved_memory', 0) / (1024**3):.2f} GB",
            ]
        )
    else:
        lines.append("")
        lines.append(
            "> GPU metrics are unavailable on this system. You can still run the CLI "
            "and CPU guides."
        )

    lines.append("")
    if tf_gpu_info and tf_gpu_info.get("devices"):
        lines.append("")
        lines.append("## TensorFlow GPU Snapshot")
        device = tf_gpu_info["devices"][0]
        lines.extend(
            [
                f"- **TF Device Name**: {device.get('name', 'Unknown')}",
                f"- **Current Memory**: {device.get('current_memory_mb', 0):.2f} MB",
                f"- **Peak Memory**: {device.get('peak_memory_mb', 0):.2f} MB",
            ]
        )

    lines.append("")
    lines.append("## Getting Started")
    lines.append("")
    lines.append("- `python -m examples.basic.pytorch_demo`")
    lines.append("- `python -m examples.basic.tensorflow_demo`")
    lines.append("- `python -m examples.cli.quickstart`")
    lines.append("")
    lines.append(
        "Need more? Visit the [Example Test Guides](docs/examples/test_guides/README.md)."
    )
    return "\n".join(lines)


def build_pytorch_stats_rows(info: dict[str, Any]) -> list[dict[str, Any]]:
    if not info:
        return []
    return [
        {
            "device": info.get("device_name", "gpu0"),
            "current": info.get("allocated_memory", 0) / (1024**2),
            "peak": info.get("max_memory_allocated", info.get("allocated_memory", 0))
            / (1024**2),
            "reserved": info.get("reserved_memory", 0) / (1024**2),
        }
    ]


def build_tensorflow_stats_rows(gpu_info: dict[str, Any]) -> list[dict[str, Any]]:
    devices = gpu_info.get("devices", []) if gpu_info else []
    rows = []
    for device in devices:
        rows.append(
            {
                "device": device.get("name", "tf-gpu"),
                "current": device.get("current_memory_mb", 0),
                "peak": device.get("peak_memory_mb", 0),
                "reserved": gpu_info.get("total_memory", 0),
            }
        )
    return rows


def build_framework_markdown(framework: str) -> str:
    if framework == "pytorch":
        return dedent(
            """
            # PyTorch Playbook

            1. **Basic profiling**
               ```bash
               python -m examples.basic.pytorch_demo
               ```
            2. **Advanced tracking (alerts, watchdog)**
               ```bash
               python -m examples.advanced.tracking_demo
               ```
            3. **Telemetry + diagnostics**
               ```bash
               python -m examples.scenarios.mps_telemetry_scenario
               python -m examples.scenarios.oom_flight_recorder_scenario --mode simulated
               gpumemprof diagnose --duration 0 --output ./artifacts/diag
               ```
            4. **CLI helpers**
               ```bash
               gpumemprof info
               gpumemprof track --duration 60 --output tracking.json
               ```

            Check the [PyTorch Testing Guide](docs/pytorch_testing_guide.md) for
            full workflows and troubleshooting steps.
            """
        ).strip()

    return dedent(
        """
        # TensorFlow Playbook

        1. **Basic profiling**
           ```bash
           python -m examples.basic.tensorflow_demo
           ```
        2. **CLI helpers**
           ```bash
           tfmemprof info
           tfmemprof monitor --duration 30 --interval 0.5
           tfmemprof track --output tf_results.json
           ```
        3. **Telemetry + diagnostics**
           ```bash
           python -m examples.scenarios.tf_end_to_end_scenario
           tfmemprof diagnose --duration 0 --output ./artifacts/tf-diag
           ```

        The [TensorFlow Testing Guide](docs/tensorflow_testing_guide.md) includes
        deeper recipes, including mixed precision and multi-GPU notes.
        """
    ).strip()


def build_cli_markdown() -> str:
    return dedent(
        """
        # CLI Quick Samples

        ```bash
        gpumemprof info
        gpumemprof monitor --duration 30 --interval 0.5
        gpumemprof track --duration 60 --output tracking.json
        gpumemprof diagnose --duration 0 --output artifacts/diag

        tfmemprof info
        tfmemprof monitor --duration 30 --interval 0.5
        tfmemprof track --duration 60 --output tf_tracking.json
        tfmemprof diagnose --duration 0 --output artifacts/tf_diag

        python -m examples.scenarios.oom_flight_recorder_scenario --mode simulated
        python -m examples.cli.capability_matrix --mode smoke --target both --oom-mode simulated --skip-tui

        # Optional: fuller dashboard
        stormlog

        # Ensure pip shows progress
        pip install --progress-bar on "stormlog[tui]"
        ```

        Use the buttons below to log summaries or copy commands.
        """
    ).strip()


def build_visual_markdown() -> str:
    return dedent(
        """
        # Visualization Tips

        - Start live tracking to collect timeline samples, then refresh the view.
        - Use `Generate PNG Plot` to save a Matplotlib graph (writes to ./visualizations).
        - Prefer `Generate HTML Plot` for an interactive Plotly view you can open in a browser.
        - A lightweight ASCII chart appears below so you can inspect trends without leaving the terminal.
        """
    ).strip()
