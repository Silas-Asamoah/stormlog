"""Command-line interface for Stormlog."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Optional, Union, cast

import psutil

from .utils import format_bytes, get_gpu_info, get_system_info, memory_summary

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
            cmd_analyze(args)
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

    if not system_info.get("cuda_available", False):
        print(f"MPS Built: {system_info.get('mps_built', False)}")
        print(f"MPS Available: {system_info.get('mps_available', False)}")
        if system_info.get("mps_available", False):
            print(
                "CUDA is not available. MPS backend is available for supported PyTorch workloads."
            )
        else:
            print("CUDA is not available. Falling back to CPU-only profiling.")
        process = psutil.Process()
        with process.oneshot():
            mem = process.memory_info()
        print(f"Process RSS: {format_bytes(mem.rss)}")
        print(f"Process VMS: {format_bytes(mem.vms)}")
        print(
            f"CPU Count: {psutil.cpu_count(logical=False)} physical / {psutil.cpu_count()} logical"
        )
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

    start_time = time.time()
    try:
        while time.time() - start_time < duration:
            # Print current status every 5 seconds
            if int((time.time() - start_time)) % 5 == 0:
                if runtime_backend in {"cuda", "rocm"} and profiler is not None:
                    torch_module = _require_torch("GPU monitoring")
                    current_mem = torch_module.cuda.memory_allocated(
                        profiler.device
                    ) / (1024**3)
                    unit = "GB"
                elif tracker is not None:
                    stats = tracker.get_statistics()
                    current_mem = stats.get("current_memory_allocated", 0) / (1024**3)
                    unit = "GB"
                else:
                    current_mem = (
                        profiler._take_snapshot().rss / (1024**2) if profiler else 0.0
                    )
                    unit = "MB"
                elapsed = time.time() - start_time
                print(
                    f"Elapsed: {elapsed:.1f}s, Current Memory: {current_mem:.2f} {unit}"
                )
            time.sleep(1)

    except KeyboardInterrupt:
        print("\nMonitoring stopped by user")

    finally:
        if tracker is not None:
            tracker.stop_tracking()
        elif profiler is not None:
            profiler.stop_monitoring()

    # Show summary
    print("\nMonitoring Summary:")
    print("-" * 30)
    if tracker is not None:
        stats = tracker.get_statistics()
        events = tracker.get_events()
        first_alloc = events[0].memory_allocated if events else 0
        last_alloc = events[-1].memory_allocated if events else 0
        summary = {
            "snapshots_collected": len(events),
            "peak_memory_usage": stats.get("peak_memory", 0),
            "memory_change_from_baseline": last_alloc - first_alloc,
        }
        unit = "GB"
        divisor = 1024**3
    else:
        summary = profiler.get_summary() if profiler is not None else {}
        unit = "GB" if gpu_runtime else "MB"
        divisor = 1024**3 if gpu_runtime else 1024**2

    print(f"Snapshots collected: {summary.get('snapshots_collected', 0)}")
    peak = summary.get("peak_memory_usage", 0)
    change = summary.get("memory_change_from_baseline", 0)
    print(f"Peak memory usage: {peak / divisor:.2f} {unit}")
    print(f"Memory change from baseline: {change / divisor:.2f} {unit}")

    # Save data if requested
    if args.output:
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
                snapshots=profiler.snapshots,
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
        )
        print("Running CPU memory tracker (no GPU backend available).")

    # Start tracking
    tracker.start_tracking()

    start_time = time.time()
    try:
        with (
            tracker.capture_oom(
                context="gpumemprof.track",
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
                    stats = tracker.get_statistics()
                    divisor = 1024**3 if gpu_runtime else 1024**2
                    unit = "GB" if gpu_runtime else "MB"
                    current_mem = stats.get("current_memory_allocated", 0) / divisor
                    peak_mem = stats.get("peak_memory", 0) / divisor
                    utilization = stats.get("memory_utilization_percent", 0)
                    print(
                        f"Elapsed: {elapsed:.1f}s, Memory: {current_mem:.2f} {unit} "
                        f"({utilization:.1f}%), Peak: {peak_mem:.2f} {unit}"
                    )

                time.sleep(1)

    except KeyboardInterrupt:
        print("\nTracking stopped by user")

    finally:
        tracker.stop_tracking()
        if gpu_runtime and tracker.last_oom_dump_path:
            print(f"OOM flight recorder dump saved to: {tracker.last_oom_dump_path}")

    # Show final statistics
    print("\nTracking Summary:")
    print("-" * 30)
    stats = tracker.get_statistics()
    divisor = 1024**3 if gpu_runtime else 1024**2
    unit = "GB" if gpu_runtime else "MB"
    print(f"Total events: {stats.get('total_events', 0)}")
    print(f"Peak memory: {stats.get('peak_memory', 0) / divisor:.2f} {unit}")

    if watchdog:
        cleanup_stats = watchdog.get_cleanup_stats()
        print(f"Automatic cleanups: {cleanup_stats.get('cleanup_count', 0)}")

    # Save events if requested
    if args.output:
        tracker.export_events(args.output, args.format)
        print(f"Events saved to: {args.output}")


def cmd_analyze(args: argparse.Namespace) -> None:
    """Handle analyze command."""
    input_file = args.input_file

    if not Path(input_file).exists():
        print(f"Error: Input file '{input_file}' not found")
        return

    print(f"Analyzing profiling results from: {input_file}")

    # Load data
    try:
        with open(input_file, "r") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading input file: {e}")
        return

    # Create analyzer
    (MemoryAnalyzer,) = _import_runtime_symbols(
        ".analyzer", ("MemoryAnalyzer",), "The analyze command"
    )
    _analyzer = MemoryAnalyzer()

    # For now, create dummy results for demonstration
    # In a real implementation, you'd parse the loaded data
    print("Analysis functionality is available through the Python API.")
    print("Please use the Python library for detailed analysis:")
    print()
    print("Example:")
    print("from gpumemprof import MemoryAnalyzer")
    print("analyzer = MemoryAnalyzer()")
    print("patterns = analyzer.analyze_memory_patterns(results)")
    print("insights = analyzer.generate_performance_insights(results)")
    print("report = analyzer.generate_optimization_report(results)")

    # Generate basic report
    print("\nBasic Analysis:")
    print(f"Input file: {input_file}")
    print(f"File size: {Path(input_file).stat().st_size} bytes")

    if "results" in data:
        print(f"Number of results: {len(data['results'])}")
    if "snapshots" in data:
        print(f"Number of snapshots: {len(data['snapshots'])}")


def cmd_diagnose(args: argparse.Namespace) -> int:
    """Produce a portable diagnostic bundle. Returns 0 (OK), 1 (failure), or 2 (memory risk)."""
    if args.duration < 0:
        print("Error: --duration must be >= 0", file=sys.stderr)
        return 1
    if args.interval <= 0:
        print("Error: --interval must be > 0", file=sys.stderr)
        return 1

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
        )
    except OSError:
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

    return int(exit_code)


if __name__ == "__main__":
    main()
