"""TensorFlow Stormlog CLI"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict

from .tf_env import configure_tensorflow_logging

configure_tensorflow_logging()

try:
    import tensorflow as tf

    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    tf = None

from stormlog.mlflow_integration import (
    add_mlflow_arguments,
    ensure_mlflow_available,
    export_diagnose_bundle_to_mlflow,
    export_tracking_run_to_mlflow,
    mlflow_config_from_namespace,
)
from stormlog.telemetry import telemetry_event_from_record, telemetry_event_to_dict
from stormlog.telemetry_sink import TelemetrySinkConfig
from stormlog.wandb_integration import (
    add_wandb_arguments,
    ensure_wandb_available,
    export_diagnose_bundle_to_wandb,
    export_tracking_run_to_wandb,
    wandb_config_from_namespace,
)

from .analyzer import MemoryAnalyzer
from .diagnose import run_diagnose
from .tracker import MemoryTracker
from .utils import format_memory, generate_summary_report, get_system_info
from .visualizer import MemoryVisualizer


def setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _normalize_telemetry_events(
    records: list[dict[str, Any]], sampling_interval_ms: int
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for record in records:
        event = telemetry_event_from_record(
            record,
            default_collector="stormlog.tensorflow.memory_tracker",
            default_sampling_interval_ms=sampling_interval_ms,
        )
        normalized.append(telemetry_event_to_dict(event))
    return normalized


def _build_telemetry_sink_config(
    args: argparse.Namespace,
) -> TelemetrySinkConfig | None:
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


def _resolve_wandb_config(args: argparse.Namespace) -> Any:
    config = wandb_config_from_namespace(args)
    if not config.enabled:
        return config
    try:
        ensure_wandb_available(config)
    except ImportError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return None
    return config


def _warn_wandb_export_failure(command_name: str, exc: Exception) -> None:
    print(f"Warning: {command_name} W&B export skipped: {exc}", file=sys.stderr)


def _resolve_mlflow_config(args: argparse.Namespace) -> Any:
    config = mlflow_config_from_namespace(args)
    if not config.enabled:
        return config
    try:
        ensure_mlflow_available(config)
    except ImportError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return None
    return config


def _warn_mlflow_export_failure(command_name: str, exc: Exception) -> None:
    print(f"Warning: {command_name} MLflow export skipped: {exc}", file=sys.stderr)


def cmd_info(args: argparse.Namespace) -> int:
    """Display system and GPU information."""
    print("TensorFlow Stormlog - System Information")
    print("=" * 50)

    system_info = get_system_info()

    print(f"Platform: {system_info['platform']}")
    print(f"Python Version: {system_info['python_version']}")
    print(f"TensorFlow Version: {system_info['tensorflow_version']}")
    print(f"CPU Count: {system_info['cpu_count']}")

    if "total_memory_gb" in system_info:
        print(f"Total System Memory: {system_info['total_memory_gb']:.2f} GB")
        print(f"Available Memory: {system_info['available_memory_gb']:.2f} GB")

    print("\nGPU Information:")
    print("-" * 20)

    gpu_info = system_info.get("gpu", {})
    backend_info = system_info.get("backend", {})
    _print_gpu_info(gpu_info, backend_info)

    if backend_info:
        print("\nTensorFlow Backend Diagnostics:")
        print("-" * 30)
        print(
            f"Hardware GPU Detected: {backend_info.get('hardware_gpu_detected', False)}"
        )
        print(f"Runtime Backend: {backend_info.get('runtime_backend', 'cpu')}")
        print(f"Runtime GPU Count: {backend_info.get('runtime_gpu_count', 0)}")
        print(f"Apple Silicon: {backend_info.get('is_apple_silicon', False)}")
        print(
            f"tensorflow-metal Installed: {backend_info.get('tensorflow_metal_installed', False)}"
        )
        print(f"CUDA Build: {backend_info.get('is_cuda_build', False)}")
        print(f"ROCm Build: {backend_info.get('is_rocm_build', False)}")
        print(f"TensorRT Build: {backend_info.get('is_tensorrt_build', False)}")

    # TensorFlow specific information
    if TF_AVAILABLE:
        print("\nTensorFlow Build Information:")
        print("-" * 30)
        try:
            build_info = tf.sysconfig.get_build_info()
            print(
                f"CUDA Build: {backend_info.get('is_cuda_build', build_info.get('is_cuda_build', 'Unknown'))}"
            )
            print(f"CUDA Version: {build_info.get('cuda_version', 'Unknown')}")
            print(f"cuDNN Version: {build_info.get('cudnn_version', 'Unknown')}")
        except Exception as e:
            print(f"Could not get build info: {e}")

    return 0


def cmd_monitor(args: argparse.Namespace) -> int:
    """Monitor GPU memory usage in real-time."""
    if not TF_AVAILABLE:
        print("Error: TensorFlow not available")
        return 1

    print("Starting TensorFlow memory monitoring...")
    print(f"Sampling interval: {args.interval} seconds")
    print(
        f"Duration: {args.duration} seconds"
        if args.duration
        else "Duration: Indefinite"
    )
    if args.threshold:
        print(f"Alert threshold: {args.threshold} MB")
    print("Press Ctrl+C to stop\n")

    tracker = MemoryTracker(
        sampling_interval=args.interval,
        alert_threshold_mb=args.threshold,
        device=args.device,
        enable_logging=True,
    )

    try:
        tracker.start_tracking()

        start_time = time.time()
        while True:
            if args.duration and (time.time() - start_time) >= args.duration:
                break

            current_memory = tracker.get_current_memory()
            print(
                f"\rCurrent memory usage: {current_memory:.1f} MB", end="", flush=True
            )

            time.sleep(1.0)  # Update display every second

    except KeyboardInterrupt:
        print("\n\nStopping monitoring...")

    finally:
        results = tracker.stop_tracking()

        _report_monitor_results(results, args)

    return 0


def cmd_track(args: argparse.Namespace) -> int:
    """Start background memory tracking."""
    if not TF_AVAILABLE:
        print("Error: TensorFlow not available")
        return 1
    wandb_config = _resolve_wandb_config(args)
    if wandb_config is None:
        return 1
    mlflow_config = _resolve_mlflow_config(args)
    if mlflow_config is None:
        return 1

    print("Starting background memory tracking...")
    job_id = getattr(args, "job_id", None)
    rank = getattr(args, "rank", None)
    local_rank = getattr(args, "local_rank", None)
    world_size = getattr(args, "world_size", None)
    telemetry_sink_config = _build_telemetry_sink_config(args)

    tracker = MemoryTracker(
        sampling_interval=args.interval,
        alert_threshold_mb=args.threshold,
        device=args.device,
        enable_logging=True,
        job_id=job_id,
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        telemetry_sink_config=telemetry_sink_config,
    )
    if telemetry_sink_config is not None:
        print(f"Append-only telemetry sink: {telemetry_sink_config.root_dir}")

    # Add alert callback
    def alert_callback(alert: Dict[str, Any]) -> None:
        print(f"\n⚠️  MEMORY ALERT: {alert['message']}")

    tracker.add_alert_callback(alert_callback)

    try:
        tracker.start_tracking()
        print("Tracking started. Press Ctrl+C to stop and save results.")

        while True:
            time.sleep(5.0)  # Check every 5 seconds

            # Show periodic updates
            stats = tracker.get_statistics()
            _print_tracking_status(stats)

    except KeyboardInterrupt:
        print("\nStopping tracking...")

    finally:
        results = tracker.stop_tracking()

        if args.output:
            sampling_interval_ms = int(round(args.interval * 1000))
            output_data = {
                "peak_memory": results.peak_memory,
                "average_memory": results.average_memory,
                "duration": results.duration,
                "memory_usage": results.memory_usage,
                "timestamps": results.timestamps,
                "alerts": results.alerts_triggered,
                "events": _normalize_telemetry_events(
                    results.events,
                    sampling_interval_ms=sampling_interval_ms,
                ),
                "history_window_limit": int(
                    getattr(results, "history_window_limit", len(results.memory_usage))
                ),
                "history_retained_samples": int(
                    getattr(
                        results, "history_retained_samples", len(results.memory_usage)
                    )
                ),
                "history_dropped_samples": int(
                    getattr(results, "history_dropped_samples", 0)
                ),
                "history_retained_events": int(
                    getattr(results, "history_retained_events", len(results.events))
                ),
                "history_dropped_events": int(
                    getattr(results, "history_dropped_events", 0)
                ),
                "history_retained_alerts": int(
                    getattr(
                        results,
                        "history_retained_alerts",
                        len(results.alerts_triggered),
                    )
                ),
                "history_dropped_alerts": int(
                    getattr(results, "history_dropped_alerts", 0)
                ),
            }

            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("w", encoding="utf-8") as f:
                json.dump(output_data, f, indent=2)

            print(f"Results saved to {args.output}")

        final_stats = tracker.get_statistics()
        _report_tracking_results(results, final_stats)

        _export_tracking_integrations(
            args, tracker, results, final_stats, wandb_config, mlflow_config
        )

    return 0


def cmd_analyze(args: argparse.Namespace) -> int:
    """Analyze profiling results."""
    if not args.input:
        print("Error: Input file required for analysis")
        return 1

    if not Path(args.input).exists():
        print(f"Error: Input file {args.input} not found")
        return 1

    print(f"Analyzing results from {args.input}...")

    # Load results
    with open(args.input, "r") as f:
        data = json.load(f)

    # Create a simple result object for analysis
    class AnalysisResult:
        def __init__(self, data: Dict[str, Any]) -> None:
            memory_usage = data.get("memory_usage") or []
            self.peak_memory_mb = data.get("peak_memory", 0)
            self.average_memory_mb = data.get("average_memory", 0)
            self.min_memory_mb = min(memory_usage, default=0)
            self.total_allocations = len(
                [
                    m
                    for i, m in enumerate(memory_usage)
                    if i > 0 and m > memory_usage[i - 1]
                ]
            )
            self.total_deallocations = len(
                [
                    m
                    for i, m in enumerate(memory_usage)
                    if i > 0 and m < memory_usage[i - 1]
                ]
            )
            self.duration = data.get("duration", 0)

            # Create fake snapshots for analysis
            self.snapshots = []
            timestamps = data.get("timestamps", list(range(len(memory_usage))))
            if len(memory_usage) != len(timestamps):
                raise ValueError(
                    "Invalid input: 'memory_usage' and 'timestamps' must have equal length"
                )

            for i, (mem, ts) in enumerate(zip(memory_usage, timestamps, strict=True)):
                snapshot = type(
                    "Snapshot",
                    (),
                    {
                        "timestamp": ts,
                        "name": f"sample_{i}",
                        "gpu_memory_mb": mem,
                        "cpu_memory_mb": 0,
                        "gpu_memory_reserved_mb": mem * 1.1,  # Estimate
                        "gpu_utilization": min(100, mem / 1000 * 100),
                        "num_tensors": 0,
                    },
                )()
                self.snapshots.append(snapshot)

    try:
        result = AnalysisResult(data)
    except ValueError as exc:
        print(f"Error: {exc}")
        return 1

    # Basic analysis
    print("\nBasic Analysis:")
    print("-" * 15)
    print(f"Peak Memory: {format_memory(result.peak_memory_mb * 1024 * 1024)}")
    print(f"Average Memory: {format_memory(result.average_memory_mb * 1024 * 1024)}")
    print(f"Duration: {result.duration:.2f} seconds")
    print(f"Memory Allocations: {result.total_allocations}")
    print(f"Memory Deallocations: {result.total_deallocations}")

    if args.detect_leaks:
        print("\nMemory Leak Analysis:")
        print("-" * 22)

        analyzer = MemoryAnalyzer()

        # Create tracking result for leak detection
        class TrackingResult:
            def __init__(self, data: Dict[str, Any]) -> None:
                self.memory_usage = data.get("memory_usage", [])
                self.timestamps = data.get("timestamps", [])
                self.memory_growth_rate = 0
                if len(self.memory_usage) > 1 and result.duration > 0:
                    self.memory_growth_rate = (
                        self.memory_usage[-1] - self.memory_usage[0]
                    ) / result.duration

        tracking_result = TrackingResult(data)
        leaks = analyzer.detect_memory_leaks(tracking_result)

        if leaks:
            print("⚠️  Potential memory leaks detected:")
            for leak in leaks:
                print(
                    f"  - {leak['type']}: {leak['description']} (Severity: {leak['severity']})"
                )
        else:
            print("✅ No memory leaks detected")

    _report_optimization(args, result)

    if args.visualize:
        print("\nGenerating visualizations...")

        visualizer = MemoryVisualizer()

        try:
            visualizer.plot_memory_timeline(result, save_path="memory_timeline.png")
            print("✅ Timeline plot saved as memory_timeline.png")
        except Exception as e:
            print(f"❌ Could not generate timeline plot: {e}")

    if args.report:
        print("\nGenerating comprehensive report...")

        report = generate_summary_report(result)

        with open(args.report, "w") as f:
            f.write(report)

        print(f"✅ Report saved to {args.report}")

    return 0


def cmd_diagnose(args: argparse.Namespace) -> int:
    """Produce a portable diagnostic bundle. Returns 0 (OK), 1 (failure), or 2 (memory risk)."""
    if args.duration < 0:
        print("Error: --duration must be >= 0", file=sys.stderr)
        return 1
    if args.interval <= 0:
        print("Error: --interval must be > 0", file=sys.stderr)
        return 1

    wandb_config = _resolve_wandb_config(args)
    if wandb_config is None:
        return 1
    mlflow_config = _resolve_mlflow_config(args)
    if mlflow_config is None:
        return 1

    command_line = " ".join(sys.argv)
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

    _print_diagnose_summary(artifact_dir, exit_code)

    if wandb_config.enabled:
        try:
            export_diagnose_bundle_to_wandb(
                wandb_config,
                command_name="tfmemprof-diagnose",
                artifact_dir=artifact_dir,
            )
            print("W&B export completed.")
        except Exception as exc:
            _warn_wandb_export_failure("tfmemprof diagnose", exc)

    if mlflow_config.enabled:
        try:
            export_diagnose_bundle_to_mlflow(
                mlflow_config,
                command_name="tfmemprof-diagnose",
                artifact_dir=artifact_dir,
            )
            print("MLflow export completed.")
        except Exception as exc:
            _warn_mlflow_export_failure("tfmemprof diagnose", exc)

    return exit_code


def _print_diagnose_summary(artifact_dir: Path, exit_code: int) -> None:
    # Structured stdout summary
    print(f"Artifact: {artifact_dir}")
    if exit_code == 0:
        status = "OK"
    elif exit_code == 2:
        status = "MEMORY_RISK"
    else:
        status = "FAILED"
    print(f"Status: {status} (exit_code={exit_code})")
    _print_diagnose_findings(artifact_dir, exit_code, status)


def _print_diagnose_findings(artifact_dir: Path, exit_code: int, status: str) -> None:
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


def main() -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="TensorFlow Stormlog CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Cookbook:
  https://stormlog.readthedocs.io/en/latest/cookbook/index.html
        """,
    )

    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose logging"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Info command
    _info_parser = subparsers.add_parser(
        "info", help="Display system and GPU information"
    )

    # Monitor command
    monitor_parser = subparsers.add_parser("monitor", help="Monitor GPU memory usage")
    monitor_parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="Sampling interval in seconds (default: 1.0)",
    )
    monitor_parser.add_argument(
        "--duration",
        type=float,
        help="Monitoring duration in seconds (default: indefinite)",
    )
    monitor_parser.add_argument(
        "--threshold", type=float, help="Memory alert threshold in MB"
    )
    monitor_parser.add_argument(
        "--device",
        default="/GPU:0",
        help="TensorFlow device to monitor (default: /GPU:0)",
    )
    monitor_parser.add_argument("--output", help="Output file for results")

    # Track command
    track_parser = subparsers.add_parser("track", help="Background memory tracking")
    track_parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="Sampling interval in seconds (default: 1.0)",
    )
    track_parser.add_argument(
        "--threshold",
        type=float,
        default=4000,
        help="Memory alert threshold in MB (default: 4000)",
    )
    track_parser.add_argument(
        "--device",
        default="/GPU:0",
        help="TensorFlow device to monitor (default: /GPU:0)",
    )
    track_parser.add_argument(
        "--job-id",
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
        "--output", required=True, help="Output file for tracking results"
    )
    track_parser.add_argument(
        "--telemetry-sink-dir",
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
    analyze_parser.add_argument(
        "--input", required=True, help="Input file with profiling results"
    )
    analyze_parser.add_argument(
        "--detect-leaks", action="store_true", help="Detect memory leaks"
    )
    analyze_parser.add_argument(
        "--optimize", action="store_true", help="Generate optimization recommendations"
    )
    analyze_parser.add_argument(
        "--visualize", action="store_true", help="Generate visualization plots"
    )
    analyze_parser.add_argument("--report", help="Generate comprehensive report file")

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
        type=str,
        default="/GPU:0",
        help="TensorFlow device to monitor (default: /GPU:0)",
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
    add_wandb_arguments(diagnose_parser)
    add_mlflow_arguments(diagnose_parser)

    args = parser.parse_args()

    setup_logging(args.verbose)

    if not args.command:
        parser.print_help()
        return 0

    # Execute command
    if args.command == "info":
        return cmd_info(args)
    elif args.command == "monitor":
        return cmd_monitor(args)
    elif args.command == "track":
        return cmd_track(args)
    elif args.command == "analyze":
        return cmd_analyze(args)
    elif args.command == "diagnose":
        return cmd_diagnose(args)
    else:
        print(f"Unknown command: {args.command}")
        return 1


def _print_gpu_info(gpu_info: Dict[str, Any], backend_info: Dict[str, Any]) -> None:
    if gpu_info.get("available", False):
        print("GPU Available: Yes")
        print(f"GPU Count: {gpu_info['count']}")
        print(
            f"Total GPU Memory: {format_memory(gpu_info['total_memory'] * 1024 * 1024)}"
        )

        for i, device in enumerate(gpu_info.get("devices", [])):
            print(f"\nGPU {i}:")
            print(f"  Name: {device.get('name', 'Unknown')}")
            print(f"  Current Memory: {device.get('current_memory_mb', 0):.1f} MB")
            print(f"  Peak Memory: {device.get('peak_memory_mb', 0):.1f} MB")
    else:
        print(
            f"GPU Hardware Detected: {'Yes' if backend_info.get('hardware_gpu_detected', False) else 'No'}"
        )
        print("GPU Available to TensorFlow Runtime: No")
        if "error" in gpu_info:
            print(f"Error: {gpu_info['error']}")
        if backend_info.get("is_apple_silicon", False) and not backend_info.get(
            "tensorflow_metal_installed", False
        ):
            print(
                "Hint: install tensorflow-metal to enable Metal GPU runtime on Apple Silicon."
            )


def _report_monitor_results(results: Any, args: argparse.Namespace) -> None:
    print("\nMonitoring Results:")
    print("-" * 20)
    print(f"Peak Memory: {results.peak_memory:.1f} MB")
    print(f"Average Memory: {results.average_memory:.1f} MB")
    print(f"Duration: {results.duration:.1f} seconds")
    print(f"Samples Collected: {len(results.memory_usage)}")
    dropped_samples = int(getattr(results, "history_dropped_samples", 0))
    if dropped_samples:
        print(f"Dropped Samples: {dropped_samples}")

    if results.alerts_triggered:
        print(f"Alerts Triggered: {len(results.alerts_triggered)}")

    if args.output:
        # Save results
        output_data = {
            "peak_memory": results.peak_memory,
            "average_memory": results.average_memory,
            "duration": results.duration,
            "memory_usage": results.memory_usage,
            "timestamps": results.timestamps,
            "alerts": results.alerts_triggered,
            "history_window_limit": int(
                getattr(results, "history_window_limit", len(results.memory_usage))
            ),
            "history_retained_samples": int(
                getattr(results, "history_retained_samples", len(results.memory_usage))
            ),
            "history_dropped_samples": int(
                getattr(results, "history_dropped_samples", 0)
            ),
            "history_retained_alerts": int(
                getattr(
                    results,
                    "history_retained_alerts",
                    len(results.alerts_triggered),
                )
            ),
            "history_dropped_alerts": int(
                getattr(results, "history_dropped_alerts", 0)
            ),
        }

        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2)

        print(f"Results saved to {args.output}")


def _print_tracking_status(stats: Dict[str, Any]) -> None:
    current_memory = stats.get("current_memory_mb")
    collector_health = str(stats.get("collector_health_status", "healthy"))
    if isinstance(current_memory, (int, float)):
        status_line = f"Current memory: {float(current_memory):.1f} MB"
    else:
        status_line = "Current memory: unavailable"
    status_line += f" | Health: {collector_health}"
    retry_at = stats.get("collector_next_retry_epoch_s")
    if isinstance(retry_at, (int, float)):
        retry_in = max(float(retry_at) - time.time(), 0.0)
        status_line += f" | Retry In: {retry_in:.1f}s"
    print(status_line)


def _report_tracking_results(results: Any, final_stats: Dict[str, Any]) -> None:
    print(f"\nTracking completed. Peak memory: {results.peak_memory:.1f} MB")
    dropped_samples = int(final_stats.get("history_dropped_samples", 0))
    dropped_events = int(final_stats.get("history_dropped_events", 0))
    if dropped_samples or dropped_events:
        print(
            "History truncation: " f"samples={dropped_samples}, events={dropped_events}"
        )
    if "collector_health_status" in final_stats:
        print(
            "Collector health: "
            f"{final_stats.get('collector_health_status', 'healthy')}"
        )
    if final_stats.get("collector_last_error"):
        print(f"Last collector error: {final_stats.get('collector_last_error')}")


def _report_optimization(args: argparse.Namespace, result: Any) -> None:
    if args.optimize:
        print("\nOptimization Analysis:")
        print("-" * 22)

        analyzer = MemoryAnalyzer()
        optimization = analyzer.score_optimization(result)

        print(f"Overall Score: {optimization['overall_score']:.1f}/10")
        print("\nCategory Scores:")
        for category, score in optimization["categories"].items():
            print(f"  {category}: {score:.1f}/10")

        if optimization["top_recommendations"]:
            print("\nTop Recommendations:")
            for i, rec in enumerate(optimization["top_recommendations"], 1):
                print(f"  {i}. {rec}")


def _export_tracking_integrations(
    args: argparse.Namespace,
    tracker: MemoryTracker,
    results: Any,
    final_stats: Dict[str, Any],
    wandb_config: Any,
    mlflow_config: Any,
) -> None:
    if wandb_config.enabled:
        try:
            export_tracking_run_to_wandb(
                wandb_config,
                command_name="tfmemprof-track",
                session_summary=tracker.get_session_summary(),
                stats=final_stats,
                events=results.events,
                output_path=args.output,
                telemetry_sink_dir=getattr(args, "telemetry_sink_dir", None),
                oom_dump_path=None,
            )
            print("W&B export completed.")
        except Exception as exc:
            _warn_wandb_export_failure("tfmemprof track", exc)

    if mlflow_config.enabled:
        try:
            export_tracking_run_to_mlflow(
                mlflow_config,
                command_name="tfmemprof-track",
                session_summary=tracker.get_session_summary(),
                stats=final_stats,
                events=results.events,
                output_path=args.output,
                telemetry_sink_dir=getattr(args, "telemetry_sink_dir", None),
                oom_dump_path=None,
            )
            print("MLflow export completed.")
        except Exception as exc:
            _warn_mlflow_export_failure("tfmemprof track", exc)


if __name__ == "__main__":
    sys.exit(main())
