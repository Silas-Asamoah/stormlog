"""Command-line interface for Stormlog."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union, cast

import psutil

try:
    from .phases import summarize_phase_resolution
except ImportError:  # pragma: no cover - phase package may land in another slice
    summarize_phase_resolution = None  # type: ignore[assignment]
from .mlflow_integration import (
    add_mlflow_arguments,
    ensure_mlflow_available,
    export_diagnose_bundle_to_mlflow,
    export_tracking_run_to_mlflow,
    mlflow_config_from_namespace,
)
from .telemetry_sink import TelemetrySinkConfig
from .utils import (
    _detect_gpu_hardware,
    format_bytes,
    get_gpu_info,
    get_system_info,
    memory_summary,
)
from .wandb_integration import (
    add_wandb_arguments,
    ensure_wandb_available,
    export_diagnose_bundle_to_wandb,
    export_tracking_run_to_wandb,
    wandb_config_from_namespace,
)

summarize_phase_resolution = cast(Any, summarize_phase_resolution)

try:
    import torch as _torch
except (
    ModuleNotFoundError
):  # pragma: no cover - exercised in torch-less subprocess tests
    _torch = cast(Any, None)

torch: Any = _torch

_TORCH_INSTALL_GUIDANCE = (
    "PyTorch is required for this feature. Install with "
    "`pip install 'stormlog[torch]'` "
    "or follow https://pytorch.org/get-started/locally/."
)
_VIZ_INSTALL_GUIDANCE = (
    "Visualization dependencies are unavailable. "
    "Install with `pip install 'stormlog[viz]'`."
)

# Stable monkeypatchable runtime hooks for tests/callers.
MemoryTracker: Any = None
MemoryWatchdog: Any = None
CPUMemoryTracker: Any = None


def _require_torch(feature: str) -> Any:
    if torch is None:
        raise ImportError(f"{feature} requires PyTorch. {_TORCH_INSTALL_GUIDANCE}")
    return torch


def _import_runtime_symbols(
    module_name: str, symbols: tuple[str, ...], feature: str
) -> tuple[Any, ...]:
    try:
        module = importlib.import_module(module_name, package=__package__)
    except ModuleNotFoundError as exc:
        if exc.name == "torch":
            raise ImportError(
                f"{feature} requires PyTorch. {_TORCH_INSTALL_GUIDANCE}"
            ) from exc
        raise
    return tuple(getattr(module, symbol) for symbol in symbols)


def _resolve_runtime_symbol(
    cache_name: str,
    module_name: str,
    symbol_name: str,
    feature: str,
) -> Any:
    cached = globals().get(cache_name)
    if cached is not None:
        return cached
    (value,) = _import_runtime_symbols(module_name, (symbol_name,), feature)
    globals()[cache_name] = value
    return value


def _is_visualization_dependency_error(exc: BaseException) -> bool:
    current: BaseException | None = exc
    visited: set[int] = set()
    message_tokens = (
        "matplotlib",
        "plotly",
        "seaborn",
        "pil",
        "pillow",
        "_imaging",
        "stormlog[viz]",
        "dlopen(",
    )

    while current is not None and id(current) not in visited:
        visited.add(id(current))
        if isinstance(current, ModuleNotFoundError) and current.name in {
            "matplotlib",
            "plotly",
            "seaborn",
            "PIL",
        }:
            return True
        if isinstance(current, (ImportError, OSError)):
            lowered = str(current).lower()
            if any(token in lowered for token in message_tokens):
                return True

        next_exc = current.__cause__
        if next_exc is None and not current.__suppress_context__:
            next_exc = current.__context__
        current = next_exc

    return False


def _build_telemetry_sink_config(
    args: argparse.Namespace,
) -> Optional[TelemetrySinkConfig]:
    sink_dir = getattr(args, "telemetry_sink_dir", None)
    if not sink_dir:
        return None
    return TelemetrySinkConfig(
        root_dir=Path(sink_dir),
        flush_every_seconds=float(getattr(args, "telemetry_flush_seconds", 2.0)),
        rollover_max_bytes=int(getattr(args, "telemetry_rollover_mb", 64))
        * 1024
        * 1024,
        retention_max_files=int(getattr(args, "telemetry_retention_files", 8)),
        retention_max_total_bytes=int(
            getattr(args, "telemetry_retention_total_mb", 512)
        )
        * 1024
        * 1024,
    )


def _resolve_wandb_config_or_exit(args: argparse.Namespace) -> Any:
    config = wandb_config_from_namespace(args)
    if not config.enabled:
        return config
    try:
        ensure_wandb_available(config)
    except ImportError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    return config


def _warn_wandb_export_failure(command_name: str, exc: Exception) -> None:
    print(f"Warning: {command_name} W&B export skipped: {exc}", file=sys.stderr)


def _resolve_mlflow_config_or_exit(args: argparse.Namespace) -> Any:
    config = mlflow_config_from_namespace(args)
    if not config.enabled:
        return config
    try:
        ensure_mlflow_available(config)
    except ImportError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    return config


def _warn_mlflow_export_failure(command_name: str, exc: Exception) -> None:
    print(f"Warning: {command_name} MLflow export skipped: {exc}", file=sys.stderr)


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Stormlog - Monitor and analyze GPU memory usage",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  gpumemprof info                          # Show GPU information
  gpumemprof monitor --duration 60         # Monitor for 60 seconds
  gpumemprof track --output tracking.csv   # Track with CSV output
  gpumemprof analyze results.json          # Analyze profiling results
  gpumemprof diagnose --output ./diag     # Produce diagnostic bundle

Cookbook:
  https://stormlog.readthedocs.io/en/latest/cookbook/index.html
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Info command
    info_parser = subparsers.add_parser("info", help="Show GPU and system information")
    info_parser.add_argument(
        "--device",
        type=int,
        default=None,
        help="GPU device ID (default: current device)",
    )
    info_parser.add_argument(
        "--detailed", action="store_true", help="Show detailed information"
    )

    # Monitor command
    monitor_parser = subparsers.add_parser("monitor", help="Monitor memory usage")
    monitor_parser.add_argument(
        "--device",
        type=int,
        default=None,
        help="GPU device ID (default: current device)",
    )
    monitor_parser.add_argument(
        "--duration",
        type=float,
        default=10.0,
        help="Monitoring duration in seconds (default: 10)",
    )
    monitor_parser.add_argument(
        "--interval",
        type=float,
        default=0.1,
        help="Sampling interval in seconds (default: 0.1)",
    )
    monitor_parser.add_argument(
        "--output", type=str, default=None, help="Output file for monitoring data"
    )
    monitor_parser.add_argument(
        "--format",
        choices=["csv", "json"],
        default="csv",
        help="Output format (default: csv)",
    )

    # Track command
    track_parser = subparsers.add_parser(
        "track", help="Real-time memory tracking with alerts"
    )
    track_parser.add_argument(
        "--device",
        type=int,
        default=None,
        help="GPU device ID (default: current device)",
    )
    track_parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Tracking duration in seconds (default: indefinite)",
    )
    track_parser.add_argument(
        "--interval",
        type=float,
        default=0.1,
        help="Sampling interval in seconds (default: 0.1)",
    )
    track_parser.add_argument(
        "--output", type=str, default=None, help="Output file for tracking events"
    )
    track_parser.add_argument(
        "--format",
        choices=["csv", "json"],
        default="csv",
        help="Output format (default: csv)",
    )
    track_parser.add_argument(
        "--watchdog", action="store_true", help="Enable automatic memory cleanup"
    )
    track_parser.add_argument(
        "--warning-threshold",
        type=float,
        default=80.0,
        help="Memory warning threshold percentage (default: 80)",
    )
    track_parser.add_argument(
        "--critical-threshold",
        type=float,
        default=95.0,
        help="Memory critical threshold percentage (default: 95)",
    )
    track_parser.add_argument(
        "--job-id",
        type=str,
        default=None,
        help="Distributed job identifier override (default: infer from env)",
    )
    track_parser.add_argument(
        "--rank",
        type=int,
        default=None,
        help="Global distributed rank override (default: infer from env)",
    )
    track_parser.add_argument(
        "--local-rank",
        type=int,
        default=None,
        help="Local distributed rank override (default: infer from env)",
    )
    track_parser.add_argument(
        "--world-size",
        type=int,
        default=None,
        help="Distributed world size override (default: infer from env)",
    )
    track_parser.add_argument(
        "--oom-flight-recorder",
        action="store_true",
        help="Enable automatic OOM flight recorder dump artifacts",
    )
    track_parser.add_argument(
        "--oom-dump-dir",
        type=str,
        default="oom_dumps",
        help="Directory used to write OOM dump bundles (default: oom_dumps)",
    )
    track_parser.add_argument(
        "--oom-buffer-size",
        type=int,
        default=None,
        help="Ring buffer size for OOM event dumps (default: max tracker events)",
    )
    track_parser.add_argument(
        "--oom-max-dumps",
        type=int,
        default=5,
        help="Maximum number of retained OOM dump bundles (default: 5)",
    )
    track_parser.add_argument(
        "--oom-max-total-mb",
        type=int,
        default=256,
        help="Maximum retained OOM dump storage in MB (default: 256)",
    )
    track_parser.add_argument(
        "--telemetry-sink-dir",
        type=str,
        default=None,
        help="Directory for append-only telemetry sink segments",
    )
    track_parser.add_argument(
        "--telemetry-flush-seconds",
        type=float,
        default=2.0,
        help="Maximum seconds between telemetry sink flushes (default: 2.0)",
    )
    track_parser.add_argument(
        "--telemetry-rollover-mb",
        type=int,
        default=64,
        help="Telemetry sink segment rollover size in MB (default: 64)",
    )
    track_parser.add_argument(
        "--telemetry-retention-files",
        type=int,
        default=8,
        help="Maximum retained telemetry sink segments (default: 8)",
    )
    track_parser.add_argument(
        "--telemetry-retention-total-mb",
        type=int,
        default=512,
        help="Maximum retained telemetry sink size in MB (default: 512)",
    )
    add_wandb_arguments(track_parser)
    add_mlflow_arguments(track_parser)

    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze profiling results")
    analyze_parser.add_argument("input_file", help="Input file with profiling results")
    analyze_parser.add_argument(
        "--output", type=str, default=None, help="Output file for analysis report"
    )
    analyze_parser.add_argument(
        "--format",
        choices=["json", "txt"],
        default="json",
        help="Output format (default: json)",
    )
    analyze_parser.add_argument(
        "--visualization", action="store_true", help="Generate visualization plots"
    )
    analyze_parser.add_argument(
        "--plot-dir",
        type=str,
        default="plots",
        help="Directory for visualization plots (default: plots)",
    )
    analyze_parser.add_argument(
        "--session-id",
        type=str,
        default=None,
        help="Explicit telemetry session id to analyze when multiple sessions are present",
    )

    # Diagnose command
    diagnose_parser = subparsers.add_parser(
        "diagnose",
        help="Produce a portable diagnostic bundle for debugging memory failures",
    )
    diagnose_parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for the artifact bundle (default: cwd)",
    )
    diagnose_parser.add_argument(
        "--device",
        type=int,
        default=None,
        help="GPU device ID (default: current device)",
    )
    diagnose_parser.add_argument(
        "--duration",
        type=float,
        default=5.0,
        help="Seconds to run tracker for telemetry (default: 5, use 0 to skip)",
    )
    diagnose_parser.add_argument(
        "--interval",
        type=float,
        default=0.5,
        help="Sampling interval for timeline (default: 0.5)",
    )
    diagnose_parser.add_argument(
        "--native-history",
        action="store_true",
        help="Capture CUDA allocator history and snapshot artifacts for debugging",
    )
    diagnose_parser.add_argument(
        "--native-history-max-entries",
        type=int,
        default=100000,
        help="Maximum CUDA allocator history entries to retain (default: 100000)",
    )
    add_wandb_arguments(diagnose_parser)
    add_mlflow_arguments(diagnose_parser)

    # Parse arguments
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    try:
        if args.command == "info":
            cmd_info(args)
        elif args.command == "monitor":
            cmd_monitor(args)
        elif args.command == "track":
            cmd_track(args)
        elif args.command == "analyze":
            sys.exit(cmd_analyze(args))
        elif args.command == "diagnose":
            sys.exit(cmd_diagnose(args))
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


def cmd_info(args: argparse.Namespace) -> None:
    """Handle info command."""
    print("Stormlog - System Information")
    print("=" * 50)

    # System info
    system_info = get_system_info()
    detected_backend = str(system_info.get("detected_backend", "cpu"))
    print(f"Platform: {system_info.get('platform', 'Unknown')}")
    print(f"Python Version: {system_info.get('python_version', 'Unknown')}")
    print(f"CUDA Available: {system_info.get('cuda_available', False)}")
    print(f"Detected Backend: {detected_backend}")

    if detected_backend == "mps":
        print(f"MPS Built: {system_info.get('mps_built', False)}")
        print(f"MPS Available: {system_info.get('mps_available', False)}")
        if system_info.get("mps_available", False):
            print(
                "CUDA is not available. MPS backend is available for supported PyTorch workloads."
            )
        _print_process_memory_info()
        return

    if not system_info.get("cuda_available", False):
        _print_cpu_runtime_info(args, system_info)
        return

    print(f"CUDA Version: {system_info.get('cuda_version', 'Unknown')}")
    if detected_backend == "rocm":
        print(f"ROCm Version: {system_info.get('rocm_version', 'Unknown')}")
    print(f"GPU Device Count: {system_info.get('cuda_device_count', 0)}")
    print(f"Current Device: {system_info.get('current_device', 0)}")
    print()

    # GPU info
    torch_module = _require_torch("The CUDA info command")
    device_id = (
        args.device if args.device is not None else torch_module.cuda.current_device()
    )
    gpu_info = get_gpu_info(device_id)

    print(f"GPU {device_id} Information:")
    print(f"  Name: {gpu_info.get('device_name', 'Unknown')}")
    print(f"  Total Memory: {gpu_info.get('total_memory', 0) / (1024**3):.2f} GB")
    print(f"  Allocated: {gpu_info.get('allocated_memory', 0) / (1024**3):.2f} GB")
    print(f"  Reserved: {gpu_info.get('reserved_memory', 0) / (1024**3):.2f} GB")
    print(f"  Multiprocessors: {gpu_info.get('multiprocessor_count', 0)}")

    if args.detailed:
        print("\nDetailed Information:")
        print("-" * 30)

        # Memory summary
        summary = memory_summary(device_id)
        print(summary)

        # Additional stats if available
        if "nvidia_smi_info" in gpu_info:
            smi_info = gpu_info["nvidia_smi_info"]
            print("\nNVIDIA-SMI Information:")
            print(f"  GPU Utilization: {smi_info.get('gpu_utilization_percent', 0)}%")
            print(f"  Temperature: {smi_info.get('temperature_c', 0)}°C")
            print(f"  Power Draw: {smi_info.get('power_draw_w', 0):.1f} W")


def cmd_monitor(args: argparse.Namespace) -> None:
    """Handle monitor command."""
    device = args.device
    duration = args.duration
    interval = args.interval

    runtime_backend = str(get_system_info().get("detected_backend", "cpu"))
    gpu_runtime = runtime_backend in {"cuda", "rocm", "mps"}

    print(f"Starting memory monitoring for {duration} seconds...")
    mode_label = f"GPU ({runtime_backend})" if gpu_runtime else "CPU"
    print(f"Mode: {mode_label}")
    print(f"Sampling interval: {interval}s")
    print("Press Ctrl+C to stop early")
    print()

    profiler, tracker = _start_monitor(runtime_backend, device, interval)

    start_time = time.time()
    try:
        while time.time() - start_time < duration:
            # Print current status every 5 seconds
            if int((time.time() - start_time)) % 5 == 0:
                current_mem_text = _monitor_memory_text(
                    runtime_backend, profiler, tracker
                )
                elapsed = time.time() - start_time
                print(f"Elapsed: {elapsed:.1f}s, Current Memory: {current_mem_text}")
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nMonitoring stopped by user")
    finally:
        if tracker is not None:
            tracker.stop_tracking()
        elif profiler is not None:
            profiler.stop_monitoring()

    _print_monitor_summary(profiler, tracker, gpu_runtime)
    if args.output:
        _export_monitor_data(args, runtime_backend, profiler, tracker)


def _start_monitor(
    runtime_backend: str, device: Any, interval: float
) -> tuple[Any, Any]:
    profiler: Optional[Any] = None
    tracker: Optional[Any] = None
    if runtime_backend in {"cuda", "rocm"}:
        (GPUMemoryProfiler,) = _import_runtime_symbols(
            ".profiler", ("GPUMemoryProfiler",), "The monitor command"
        )
        profiler = GPUMemoryProfiler(device=device)
        profiler.start_monitoring(interval)
    elif runtime_backend == "mps":
        (MemoryTracker,) = _import_runtime_symbols(
            ".tracker", ("MemoryTracker",), "The monitor command"
        )
        if device is not None:
            print("Ignoring --device for MPS runtime (single logical device).")
        tracker = MemoryTracker(
            device="mps",
            sampling_interval=interval,
            enable_alerts=False,
        )
        tracker.start_tracking()
    else:
        (CPUMemoryProfiler,) = _import_runtime_symbols(
            ".cpu_profiler", ("CPUMemoryProfiler",), "The monitor command"
        )
        profiler = CPUMemoryProfiler()
        profiler.start_monitoring(interval)

    return profiler, tracker


def _monitor_memory_text(runtime_backend: str, profiler: Any, tracker: Any) -> str:
    if runtime_backend in {"cuda", "rocm"} and profiler is not None:
        torch_module = _require_torch("GPU monitoring")
        current_mem = torch_module.cuda.memory_allocated(profiler.device) / (1024**3)
        return f"{current_mem:.2f} GB"
    if tracker is not None:
        stats = tracker.get_statistics()
        current_allocated = stats.get("current_memory_allocated")
        return (
            f"{float(current_allocated) / (1024**3):.2f} GB"
            if isinstance(current_allocated, (int, float))
            else "-"
        )
    current_mem = profiler._take_snapshot().rss / (1024**2) if profiler else 0.0
    return f"{current_mem:.2f} MB"


def _tracker_monitor_summary(tracker: Any) -> dict[str, Any]:
    stats = tracker.get_statistics()
    events = tracker.get_events()
    allocations = [
        event.memory_allocated
        for event in events
        if event.event_type == "sample" and event.memory_allocated is not None
    ]
    allocator_change = allocations[-1] - allocations[0] if allocations else None
    return {
        "snapshots_collected": len(events),
        "peak_memory_usage": stats.get("peak_memory"),
        "memory_change_from_baseline": allocator_change,
        "peak_device_usage": stats.get("peak_device_used"),
    }


def _print_monitor_summary(profiler: Any, tracker: Any, gpu_runtime: bool) -> None:
    print("\nMonitoring Summary:")
    print("-" * 30)
    if tracker is not None:
        summary = _tracker_monitor_summary(tracker)
        unit = "GB"
        divisor = 1024**3
    else:
        summary = profiler.get_summary() if profiler is not None else {}
        unit = "GB" if gpu_runtime else "MB"
        divisor = 1024**3 if gpu_runtime else 1024**2

    print(f"Snapshots collected: {summary.get('snapshots_collected', 0)}")
    peak = summary.get("peak_memory_usage")
    change = summary.get("memory_change_from_baseline")
    print(
        "Peak memory usage: "
        + (f"{peak / divisor:.2f} {unit}" if isinstance(peak, (int, float)) else "N/A")
    )
    print(
        "Memory change from baseline: "
        + (
            f"{change / divisor:.2f} {unit}"
            if isinstance(change, (int, float))
            else "N/A"
        )
    )
    peak_device = summary.get("peak_device_usage")
    if isinstance(peak_device, (int, float)):
        print(f"Peak device usage: {peak_device / divisor:.2f} {unit}")


def _export_monitor_data(
    args: argparse.Namespace, runtime_backend: str, profiler: Any, tracker: Any
) -> None:
    if runtime_backend in {"cuda", "rocm"} and profiler is not None:
        try:
            from .visualizer import MemoryVisualizer
        except ImportError:
            print(
                "Visualization export requires optional dependencies. "
                "Install with `pip install stormlog[viz]`."
            )
            return

        visualizer = MemoryVisualizer(profiler)
        output_path = visualizer.export_data(
            snapshots=list(profiler.snapshots),
            format=args.format,
            save_path=Path(args.output).stem,
        )
        print(f"Data saved to: {output_path}")
    elif tracker is not None:
        tracker.export_events(args.output, args.format)
        print(f"Events saved to: {args.output}")
    else:
        print(
            "Skipping visualization export: CPU monitoring snapshots are not supported by MemoryVisualizer."
        )


def cmd_track(args: argparse.Namespace) -> None:
    """Handle track command."""
    wandb_config = _resolve_wandb_config_or_exit(args)
    mlflow_config = _resolve_mlflow_config_or_exit(args)
    tracker, watchdog, runtime_backend = _build_tracking_runtime(args)
    gpu_runtime = runtime_backend in {"cuda", "rocm", "mps"}
    _run_tracking_loop(tracker, args.duration, runtime_backend)
    stats = _print_tracking_summary(tracker, watchdog, gpu_runtime)

    if args.output:
        tracker.export_events(args.output, args.format)
        print(f"Events saved to: {args.output}")

    _export_tracking_integrations(args, tracker, stats, wandb_config, mlflow_config)


def _build_tracking_runtime(args: argparse.Namespace) -> tuple[Any, Any, str]:
    device = args.device
    duration = args.duration
    interval = args.interval
    job_id = getattr(args, "job_id", None)
    rank = getattr(args, "rank", None)
    local_rank = getattr(args, "local_rank", None)
    world_size = getattr(args, "world_size", None)

    print("Starting real-time memory tracking...")
    print(f"Device: {device if device is not None else 'current'}")
    print(f"Sampling interval: {interval}s")
    print(f"Duration: {duration}s" if duration else "Duration: indefinite")
    print("Press Ctrl+C to stop")
    print()

    runtime_backend = str(get_system_info().get("detected_backend", "cpu"))
    gpu_runtime = runtime_backend in {"cuda", "rocm", "mps"}
    telemetry_sink_config = _build_telemetry_sink_config(args)
    if telemetry_sink_config is not None:
        print(f"Append-only telemetry sink: {telemetry_sink_config.root_dir}")
    tracker: Any
    watchdog: Optional[Any] = None
    if gpu_runtime:
        tracker_cls = _resolve_runtime_symbol(
            "MemoryTracker",
            ".tracker",
            "MemoryTracker",
            "The track command",
        )
        tracker_device: Optional[Union[str, int]]
        if runtime_backend == "mps":
            if device is not None:
                print("Ignoring --device for MPS runtime (single logical device).")
            tracker_device = "mps"
        else:
            tracker_device = device
        tracker = tracker_cls(
            device=tracker_device,
            sampling_interval=interval,
            enable_alerts=True,
            enable_oom_flight_recorder=args.oom_flight_recorder,
            oom_dump_dir=args.oom_dump_dir,
            oom_buffer_size=args.oom_buffer_size,
            oom_max_dumps=args.oom_max_dumps,
            oom_max_total_mb=args.oom_max_total_mb,
            job_id=job_id,
            rank=rank,
            local_rank=local_rank,
            world_size=world_size,
            telemetry_sink_config=telemetry_sink_config,
        )

        if args.oom_flight_recorder:
            print("OOM flight recorder enabled:")
            print(f"  Dump directory: {args.oom_dump_dir}")
            buffer_value = tracker.oom_buffer_size
            print(f"  Buffer size: {buffer_value} events")
            print(f"  Max dumps: {args.oom_max_dumps}")
            print(f"  Max total size: {args.oom_max_total_mb} MB")

        # Set thresholds
        tracker.set_threshold("memory_warning_percent", args.warning_threshold)
        tracker.set_threshold("memory_critical_percent", args.critical_threshold)

        # Add alert callback
        def alert_callback(event: Any) -> None:
            timestamp = time.strftime("%H:%M:%S", time.localtime(event.timestamp))
            print(f"[{timestamp}] {event.event_type.upper()}: {event.context}")

        tracker.add_alert_callback(alert_callback)

        # Create watchdog if requested
        if args.watchdog:
            watchdog_cls = _resolve_runtime_symbol(
                "MemoryWatchdog",
                ".tracker",
                "MemoryWatchdog",
                "The track command",
            )
            watchdog = watchdog_cls(tracker)
            print("Memory watchdog enabled - automatic cleanup activated")
    else:
        cpu_tracker_cls = _resolve_runtime_symbol(
            "CPUMemoryTracker",
            ".cpu_profiler",
            "CPUMemoryTracker",
            "The track command",
        )
        tracker = cpu_tracker_cls(
            sampling_interval=interval,
            job_id=job_id,
            rank=rank,
            local_rank=local_rank,
            world_size=world_size,
            telemetry_sink_config=telemetry_sink_config,
        )
        print("Running CPU memory tracker (no GPU backend available).")

    return tracker, watchdog, runtime_backend


def _run_tracking_loop(
    tracker: Any, duration: float | None, runtime_backend: str
) -> None:
    gpu_runtime = runtime_backend in {"cuda", "rocm", "mps"}
    tracker.start_tracking()

    start_time = time.time()
    try:
        with (
            tracker.capture_oom(
                context="stormlog.track",
                metadata={"command": "track", "runtime_backend": runtime_backend},
            )
            if gpu_runtime
            else nullcontext()
        ):
            while True:
                elapsed = time.time() - start_time

                # Check duration limit
                if duration and elapsed >= duration:
                    break

                # Print status every 10 seconds
                if int(elapsed) % 10 == 0:
                    _print_tracking_status(tracker, elapsed, gpu_runtime)

                time.sleep(1)

    except KeyboardInterrupt:
        print("\nTracking stopped by user")

    finally:
        tracker.stop_tracking()
        if gpu_runtime and tracker.last_oom_dump_path:
            print(f"OOM flight recorder dump saved to: {tracker.last_oom_dump_path}")


def _print_tracking_status(tracker: Any, elapsed: float, gpu_runtime: bool) -> None:
    stats = tracker.get_statistics()
    divisor = 1024**3 if gpu_runtime else 1024**2
    unit = "GB" if gpu_runtime else "MB"
    current_allocated = stats.get("current_memory_allocated")
    peak_memory = stats.get("peak_memory")
    utilization = stats.get("memory_utilization_percent")
    collector_health = str(stats.get("collector_health_status", "healthy"))
    retry_at = stats.get("collector_next_retry_epoch_s")
    current_mem_text = (
        f"{float(current_allocated) / divisor:.2f} {unit}"
        if isinstance(current_allocated, (int, float))
        else "-"
    )
    peak_mem_text = (
        f"{float(peak_memory) / divisor:.2f} {unit}"
        if isinstance(peak_memory, (int, float))
        else "N/A"
    )
    utilization_text = (
        f"{float(utilization):.1f}%" if isinstance(utilization, (int, float)) else "-"
    )
    status_line = (
        f"Elapsed: {elapsed:.1f}s, Memory: {current_mem_text} "
        f"({utilization_text}), Peak: {peak_mem_text}, "
        f"Health: {collector_health}"
    )
    if isinstance(retry_at, (int, float)):
        retry_in = max(float(retry_at) - time.time(), 0.0)
        status_line += f", Retry In: {retry_in:.1f}s"
    print(status_line)


def _print_tracking_summary(
    tracker: Any, watchdog: Any, gpu_runtime: bool
) -> dict[str, Any]:
    print("\nTracking Summary:")
    print("-" * 30)
    stats = tracker.get_statistics()
    divisor = 1024**3 if gpu_runtime else 1024**2
    unit = "GB" if gpu_runtime else "MB"
    print(f"Total events: {stats.get('total_events', 0)}")
    peak_memory = stats.get("peak_memory")
    peak_memory_text = (
        f"{float(peak_memory) / divisor:.2f} {unit}"
        if isinstance(peak_memory, (int, float))
        else "N/A"
    )
    print(f"Peak memory: {peak_memory_text}")
    peak_device = stats.get("peak_device_used")
    if isinstance(peak_device, (int, float)):
        print(f"Peak device memory: {peak_device / divisor:.2f} {unit}")
    if "collector_health_status" in stats:
        print(f"Collector health: {stats.get('collector_health_status', 'healthy')}")
    if stats.get("collector_last_error"):
        print(f"Last collector error: {stats.get('collector_last_error')}")

    if watchdog:
        cleanup_stats = watchdog.get_cleanup_stats()
        print(f"Automatic cleanups: {cleanup_stats.get('cleanup_count', 0)}")

    return cast(dict[str, Any], stats)


def _export_tracking_integrations(
    args: argparse.Namespace,
    tracker: Any,
    stats: dict[str, Any],
    wandb_config: Any,
    mlflow_config: Any,
) -> None:
    if wandb_config.enabled:
        try:
            export_tracking_run_to_wandb(
                wandb_config,
                command_name="gpumemprof-track",
                session_summary=tracker.get_session_summary(),
                stats=stats,
                events=tracker.get_events(),
                output_path=args.output,
                telemetry_sink_dir=getattr(args, "telemetry_sink_dir", None),
                oom_dump_path=getattr(tracker, "last_oom_dump_path", None),
            )
            print("W&B export completed.")
        except Exception as exc:
            _warn_wandb_export_failure("gpumemprof track", exc)

    if mlflow_config.enabled:
        try:
            export_tracking_run_to_mlflow(
                mlflow_config,
                command_name="gpumemprof-track",
                session_summary=tracker.get_session_summary(),
                stats=stats,
                events=tracker.get_events(),
                output_path=args.output,
                telemetry_sink_dir=getattr(args, "telemetry_sink_dir", None),
                oom_dump_path=getattr(tracker, "last_oom_dump_path", None),
            )
            print("MLflow export completed.")
        except Exception as exc:
            _warn_mlflow_export_failure("gpumemprof track", exc)


def _json_default(value: Any) -> Any:
    """Convert common non-JSON-native values to plain Python scalars."""

    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _input_artifact_size_bytes(path: Path) -> int:
    if path.is_file():
        return int(path.stat().st_size)
    if path.is_dir():
        return sum(entry.stat().st_size for entry in path.rglob("*") if entry.is_file())
    return 0


def _phase_summary_from_payload(payload: Any) -> str | None:
    if not isinstance(payload, dict):
        return None
    explicit_summary = _explicit_phase_summary(payload.get("phase_summary"))
    if explicit_summary is not None:
        return explicit_summary
    if summarize_phase_resolution is None:
        return None  # type: ignore[unreachable]
    phase_path = payload.get("phase_path")
    phase_paths = payload.get("phase_paths")
    normalized_phase_path = phase_path if isinstance(phase_path, str) else None
    normalized_phase_paths = (
        [str(path) for path in phase_paths if str(path)]
        if isinstance(phase_paths, list)
        else None
    )
    return summarize_phase_resolution(
        phase_resolution=payload.get("phase_resolution"),
        phase_path=normalized_phase_path,
        phase_paths=normalized_phase_paths,
    )


def _build_analyze_summary(
    input_file: str, file_size_bytes: int, report: dict[str, Any]
) -> str:
    """Create a concise human-readable analyze summary."""

    lines = [
        f"Analyzing profiling results from: {input_file}",
        "",
        "Basic Analysis:",
        f"Input file: {input_file}",
        f"File size: {file_size_bytes} bytes",
    ]

    lines.extend(_gap_summary_lines(report))

    availability = report.get("analysis_availability")
    if isinstance(availability, dict):
        unavailable_reasons = {
            str(item.get("reason"))
            for item in availability.values()
            if isinstance(item, dict)
            and item.get("available") is False
            and item.get("reason")
        }
        for reason in sorted(unavailable_reasons):
            lines.append(f"Capability warning: {reason}")

    cross_rank_analysis = report.get("cross_rank_analysis")
    if isinstance(cross_rank_analysis, dict):
        lines.extend(_cross_rank_summary_lines(cross_rank_analysis))
    else:
        notes = report.get("notes", [])
        if notes:
            lines.append("Notes: " + " ".join(str(note) for note in notes))

    return "\n".join(lines)


def _json_payload_looks_like_telemetry(payload: Any) -> bool:
    """Return whether a loaded JSON payload plausibly contains telemetry events."""

    candidate_keys = {
        "schema_version",
        "timestamp",
        "timestamp_ns",
        "event_type",
        "memory_allocated",
        "memory_mb",
        "allocator_allocated_bytes",
        "device_used_bytes",
    }

    if isinstance(payload, dict):
        if isinstance(payload.get("events"), list):
            return True
        return any(key in payload for key in candidate_keys)

    if isinstance(payload, list):
        return any(
            isinstance(item, dict) and any(key in item for key in candidate_keys)
            for item in payload
        )

    return False


@dataclass(frozen=True)
class _AnalysisInput:
    events: list[Any] | None
    sessions: list[Any]
    session_note: str | None = None
    data: Any = None


def _load_analysis_input(
    input_path: Path, requested_session_id: str | None
) -> _AnalysisInput | None:
    (load_telemetry_sessions,) = _import_runtime_symbols(
        ".telemetry", ("load_telemetry_sessions",), "The analyze command"
    )

    try:
        loaded_sessions = load_telemetry_sessions(input_path, permissive_legacy=True)
        if loaded_sessions:
            selected_session = _select_analysis_session(
                loaded_sessions, requested_session_id
            )
            if selected_session is None:
                return None
            events = list(selected_session.events)
            session_note = (
                "Telemetry session selected: "
                f"{selected_session.summary.session_id} "
                f"({selected_session.summary.status}, "
                f"started_at_ns={selected_session.summary.started_at_ns})."
            )
            return _AnalysisInput(events, loaded_sessions, session_note)
        return _AnalysisInput([], loaded_sessions)
    except Exception as exc:
        return _load_analysis_json_fallback(input_path, exc)


def _select_analysis_session(
    loaded_sessions: list[Any], requested_session_id: str | None
) -> Any:
    if requested_session_id is None:
        return loaded_sessions[0]
    selected_session = next(
        (
            loaded
            for loaded in loaded_sessions
            if loaded.summary.session_id == requested_session_id
        ),
        None,
    )
    if selected_session is None:
        print(
            "Error parsing telemetry events: Requested session_id not found: "
            f"{requested_session_id}"
        )
    return selected_session


def _load_analysis_json_fallback(
    input_path: Path, telemetry_error: Exception
) -> _AnalysisInput | None:
    if input_path.is_dir() or input_path.suffix.lower() == ".jsonl":
        print(f"Error parsing telemetry events: {telemetry_error}")
        return None
    try:
        with input_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception as error:
        print(f"Error loading input file: {error}")
        return None
    if _json_payload_looks_like_telemetry(data):
        print(f"Error parsing telemetry events: {telemetry_error}")
        return None
    return _AnalysisInput(None, [], data=data)


def _generate_analysis_report(
    analysis_input: _AnalysisInput, requested_session_id: str | None
) -> dict[str, Any]:
    if analysis_input.events is not None:
        (MemoryAnalyzer,) = _import_runtime_symbols(
            ".analyzer", ("MemoryAnalyzer",), "The analyze command"
        )
        analyzer = MemoryAnalyzer()
        report = analyzer.generate_optimization_report(events=analysis_input.events)
        if analysis_input.session_note:
            report["session"] = {
                "selected_session_id": (
                    requested_session_id
                    if requested_session_id is not None
                    else (
                        analysis_input.sessions[0].summary.session_id
                        if analysis_input.sessions
                        else None
                    )
                ),
                "discovered_session_ids": [
                    loaded.summary.session_id for loaded in analysis_input.sessions
                ],
            }
    else:
        data = analysis_input.data
        report = {
            "summary": {
                "analysis_timestamp": None,
                "total_functions_analyzed": 0,
                "total_function_calls": 0,
                "total_memory_allocated": 0,
                "total_execution_time": 0,
            },
            "top_level_keys": sorted(data.keys()) if isinstance(data, dict) else [],
            "notes": ["JSON payload does not contain telemetry events"],
        }
    return cast(dict[str, Any], report)


def cmd_analyze(args: argparse.Namespace) -> int:
    """Handle analyze command."""
    input_file = args.input_file
    input_path = Path(input_file)

    if not input_path.exists():
        print(f"Error: Input file '{input_file}' not found")
        return 1

    requested_session_id = getattr(args, "session_id", None)
    analysis_input = _load_analysis_input(input_path, requested_session_id)
    if analysis_input is None:
        return 1
    report = _generate_analysis_report(analysis_input, requested_session_id)

    summary_text = _build_analyze_summary(
        input_file=input_file,
        file_size_bytes=_input_artifact_size_bytes(input_path),
        report=report,
    )
    print(summary_text)
    if analysis_input.session_note:
        print(analysis_input.session_note)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if args.format == "json":
            output_path.write_text(
                json.dumps(report, indent=2, default=_json_default) + "\n",
                encoding="utf-8",
            )
        else:
            output_path.write_text(summary_text + "\n", encoding="utf-8")
        print(f"Analysis report saved to: {output_path}")

    if args.visualization:
        _render_analysis_visualization(args, analysis_input.events, report)

    return 0


def _render_analysis_visualization(
    args: argparse.Namespace, events: list[Any] | None, report: dict[str, Any]
) -> None:
    if events is None or "cross_rank_analysis" not in report:
        print(
            "Visualization skipped: cross-rank plots require multi-rank telemetry input."
        )
        return
    try:
        (MemoryVisualizer,) = _import_runtime_symbols(
            ".visualizer", ("MemoryVisualizer",), "The analyze command"
        )
        plot_dir = Path(args.plot_dir)
        plot_dir.mkdir(parents=True, exist_ok=True)
        plot_path = plot_dir / "cross_rank_timeline.png"
        MemoryVisualizer().plot_cross_rank_timeline(
            events=events, save_path=str(plot_path)
        )
        print(f"Visualization saved to: {plot_path}")
    except Exception as exc:
        if _is_visualization_dependency_error(exc):
            print(f"Visualization skipped: {_VIZ_INSTALL_GUIDANCE}")
        else:
            print(f"Visualization skipped: {exc}")


def cmd_diagnose(args: argparse.Namespace) -> int:
    """Produce a portable diagnostic bundle. Returns 0 (OK), 1 (failure), or 2 (memory risk)."""
    if not _valid_diagnose_arguments(args):
        return 1

    wandb_config = _resolve_wandb_config_or_exit(args)
    mlflow_config = _resolve_mlflow_config_or_exit(args)
    command_line = " ".join(sys.argv)
    (run_diagnose,) = _import_runtime_symbols(
        ".diagnose", ("run_diagnose",), "The diagnose command"
    )
    try:
        artifact_dir, exit_code = run_diagnose(
            output=args.output,
            device=args.device,
            duration=args.duration,
            interval=args.interval,
            command_line=command_line,
            native_history=getattr(args, "native_history", False),
            native_history_max_entries=getattr(
                args,
                "native_history_max_entries",
                100000,
            ),
        )
    except (OSError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    # Structured stdout summary
    print(f"Artifact: {artifact_dir}")
    if exit_code == 0:
        status = "OK"
    elif exit_code == 2:
        status = "MEMORY_RISK"
    else:
        status = "FAILED"
    print(f"Status: {status} (exit_code={exit_code})")

    _print_diagnostic_findings(artifact_dir, exit_code, status)

    if wandb_config.enabled:
        try:
            export_diagnose_bundle_to_wandb(
                wandb_config,
                command_name="gpumemprof-diagnose",
                artifact_dir=artifact_dir,
            )
            print("W&B export completed.")
        except Exception as exc:
            _warn_wandb_export_failure("gpumemprof diagnose", exc)

    if mlflow_config.enabled:
        try:
            export_diagnose_bundle_to_mlflow(
                mlflow_config,
                command_name="gpumemprof-diagnose",
                artifact_dir=artifact_dir,
            )
            print("MLflow export completed.")
        except Exception as exc:
            _warn_mlflow_export_failure("gpumemprof diagnose", exc)

    return int(exit_code)


def _print_cpu_runtime_info(
    args: argparse.Namespace, system_info: dict[str, Any]
) -> None:
    print(f"MPS Built: {system_info.get('mps_built', False)}")
    print(f"MPS Available: {system_info.get('mps_available', False)}")
    hardware_info = _detect_gpu_hardware()
    devices = hardware_info.get("devices", [])

    print(
        "GPU Hardware Detected: "
        f"{'Yes' if hardware_info.get('hardware_gpu_detected', False) else 'No'}"
    )
    if args.device is not None:
        print("Ignoring --device because no supported PyTorch GPU runtime is active.")

    if devices:
        print("Detected GPU Hardware:")
        for device in devices:
            print(f"  {device.get('name', 'Unknown')}")
    print("GPU Available to PyTorch Runtime: No")

    if devices:
        print(
            "Supported PyTorch GPU runtimes: NVIDIA CUDA, AMD ROCm-backed "
            "PyTorch on Linux, Apple MPS."
        )
        if args.detailed:
            print("\nHardware Probe Details:")
            print("-" * 30)
            for index, device in enumerate(devices):
                print(f"  Device {index}: {device.get('name', 'Unknown')}")
                print(f"    Vendor: {device.get('vendor', 'unknown')}")
                print(f"    Source: {device.get('source', 'unknown')}")
    else:
        print("CUDA is not available. Falling back to CPU-only profiling.")

    _print_process_memory_info()


def _explicit_phase_summary(phase_summary: Any) -> str | None:
    if isinstance(phase_summary, dict):
        summary_path = phase_summary.get("phase_path")
        summary_source = phase_summary.get("source")
        if isinstance(summary_path, str) and summary_path:
            if summary_source == "heuristic":
                return f"(likely) {summary_path}"
            return summary_path
    return None


def _gap_summary_lines(report: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    if "gap_analysis" in report:
        lines.append(f"Gap findings: {len(report['gap_analysis'])}")
        if report["gap_analysis"]:
            top_gap_phase = next(
                (
                    _phase_summary_from_payload(item.get("phase_attribution"))
                    for item in report["gap_analysis"]
                    if isinstance(item, dict)
                    and _phase_summary_from_payload(item.get("phase_attribution"))
                ),
                None,
            )
            if top_gap_phase:
                lines.append(f"Top gap phase: {top_gap_phase}")

    return lines


def _cross_rank_summary_lines(cross_rank_analysis: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    participating_ranks = cross_rank_analysis.get("participating_ranks", [])
    missing_ranks = cross_rank_analysis.get("missing_ranks", [])
    suspects = cross_rank_analysis.get("first_cause_suspects", [])
    lines.extend(
        [
            "",
            "Distributed Analysis:",
            "Participating ranks: "
            + (", ".join(str(rank) for rank in participating_ranks) or "none"),
            "Missing ranks: "
            + (", ".join(str(rank) for rank in missing_ranks) or "none"),
        ]
    )

    cluster_onset = cross_rank_analysis.get("cluster_onset_timestamp_ns")
    if cluster_onset is not None:
        lines.append(f"Cluster onset (aligned ns): {cluster_onset}")

    if suspects:
        top_suspect = suspects[0]
        lines.extend(
            [
                "Top first-cause suspect: "
                f"rank {top_suspect['rank']} ({top_suspect['confidence']})",
                "Evidence: "
                f"timestamp_ns={top_suspect['first_spike_timestamp_ns']}, "
                f"aligned_timestamp_ns={top_suspect['aligned_first_spike_timestamp_ns']}, "
                f"lead_ns={top_suspect['lead_over_cluster_onset_ns']}, "
                f"delta={format_bytes(int(top_suspect['peak_delta_bytes']))}",
            ]
        )
        top_suspect_phase = _phase_summary_from_payload(
            top_suspect.get("phase_attribution")
        )
        if top_suspect_phase:
            lines.append(f"Suspect phase: {top_suspect_phase}")
    else:
        lines.append("No qualifying first-cause suspect identified.")

    notes = cross_rank_analysis.get("notes", [])
    if notes:
        lines.append("Notes: " + " ".join(str(note) for note in notes))
    return lines


def _valid_diagnose_arguments(args: argparse.Namespace) -> bool:
    if args.duration < 0:
        print("Error: --duration must be >= 0", file=sys.stderr)
        return False
    if args.interval <= 0:
        print("Error: --interval must be > 0", file=sys.stderr)
        return False
    if getattr(args, "native_history_max_entries", 100000) <= 0:
        print("Error: --native-history-max-entries must be > 0", file=sys.stderr)
        return False

    return True


def _print_diagnostic_findings(artifact_dir: Path, exit_code: int, status: str) -> None:
    # One-line findings from manifest/summary
    try:
        manifest_path = artifact_dir / "manifest.json"
        if manifest_path.exists():
            with open(manifest_path) as f:
                manifest = json.load(f)
            if manifest.get("risk_detected"):
                summary_path = artifact_dir / "diagnostic_summary.json"
                if summary_path.exists():
                    with open(summary_path) as f:
                        summary = json.load(f)
                    flags = summary.get("risk_flags", {})
                    parts = [k for k, v in flags.items() if v]
                    if parts:
                        print(f"Findings: {', '.join(parts)}")
        if exit_code == 0 and status == "OK":
            print("Findings: no memory risk detected")
    except (OSError, json.JSONDecodeError):
        pass


def _print_process_memory_info() -> None:
    process = psutil.Process()
    with process.oneshot():
        mem = process.memory_info()
    print(f"Process RSS: {format_bytes(mem.rss)}")
    print(f"Process VMS: {format_bytes(mem.vms)}")
    print(
        f"CPU Count: {psutil.cpu_count(logical=False)} physical / {psutil.cpu_count()} logical"
    )


if __name__ == "__main__":
    main()
