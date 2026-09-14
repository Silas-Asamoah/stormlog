"""JAX Stormlog CLI"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, Mapping, Sequence

from stormlog.telemetry import telemetry_event_from_record, telemetry_event_to_dict
from stormlog.telemetry_sink import TelemetrySinkConfig

if TYPE_CHECKING:
    from stormlog._mlflow.core import MlflowExportConfig
    from stormlog.session import SessionSummary

    from .tracker import MemoryTracker

try:
    from stormlog.wandb_integration import (
        add_wandb_arguments,
        ensure_wandb_available,
        export_diagnose_bundle_to_wandb,
        export_tracking_run_to_wandb,
        wandb_config_from_namespace,
    )

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

    def add_wandb_arguments(parser: Any) -> None:
        pass

    def wandb_config_from_namespace(args: Any) -> Any:  # type: ignore[misc]
        class DummyConfig:
            enabled = False

        return DummyConfig()


try:
    from stormlog.mlflow_integration import (
        add_mlflow_arguments,
        ensure_mlflow_available,
        export_diagnose_bundle_to_mlflow,
        export_tracking_run_to_mlflow,
        mlflow_config_from_namespace,
    )

    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

    def add_mlflow_arguments(parser: Any) -> None:
        pass

    def mlflow_config_from_namespace(args: Any) -> Any:  # type: ignore[misc]
        class DummyConfig:
            enabled = bool(getattr(args, "mlflow", False))

        return DummyConfig()

    def ensure_mlflow_available(config: MlflowExportConfig) -> None:
        pass

    def export_tracking_run_to_mlflow(
        config: MlflowExportConfig,
        *,
        command_name: str,
        session_summary: SessionSummary | None,
        stats: Mapping[str, Any],
        events: Sequence[Any],
        output_path: str | Path | None = None,
        telemetry_sink_dir: str | Path | None = None,
        oom_dump_path: str | Path | None = None,
        attribution_bundle_dir: str | Path | None = None,
    ) -> None:
        pass

    def export_diagnose_bundle_to_mlflow(
        config: MlflowExportConfig,
        *,
        command_name: str,
        artifact_dir: str | Path,
    ) -> None:
        pass


from .jax_env import configure_jax_logging
from .utils import format_memory, get_system_info

configure_jax_logging()

jax: Any
try:
    import jax as _jax

    jax = _jax
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    jax = None


def _load_memory_tracker() -> type[MemoryTracker]:
    """Load the JAX tracker only after a command confirms JAX is available."""
    from .tracker import MemoryTracker

    return MemoryTracker


def _load_run_diagnose() -> Callable[..., tuple[Path, int]]:
    """Load the diagnose entry point only for a guarded diagnose command."""
    from .diagnose import run_diagnose

    return run_diagnose


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
            default_collector="stormlog.jax.memory_tracker",
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


def _tracking_memory_capability(results: Any) -> Dict[str, Any]:
    """Return JSON-safe capability fields for current and legacy results."""
    available = getattr(results, "device_memory_available", True)
    source = getattr(results, "memory_source", "jax_device_stats")
    reason = getattr(results, "device_memory_unavailable_reason", None)
    process_memory = getattr(results, "process_memory_bytes", None)
    return {
        "device_memory_available": available if isinstance(available, bool) else True,
        "memory_source": source if isinstance(source, str) else "jax_device_stats",
        "device_memory_unavailable_reason": reason if isinstance(reason, str) else None,
        "process_memory_bytes": (
            process_memory if isinstance(process_memory, int) else None
        ),
    }


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
    """Display system and device information."""
    print("JAX Stormlog - System Information")
    print("=" * 50)

    system_info = get_system_info()

    print(f"Platform: {system_info['platform']}")
    print(f"Python Version: {system_info['python_version']}")
    print(f"CPU Count: {system_info['cpu_count']}")

    if "total_memory_gb" in system_info:
        print(f"Total System Memory: {system_info['total_memory_gb']:.2f} GB")
        if "available_memory_gb" in system_info:
            print(f"Available Memory: {system_info['available_memory_gb']:.2f} GB")

    backend_info = system_info.get("backend", {})
    device_count = backend_info.get("device_count", 0)

    print("\nJAX Backend Information:")
    print("-" * 30)
    print(f"Runtime Backend: {backend_info.get('runtime_backend', 'Unknown')}")
    print(f"Raw JAX Backend: {backend_info.get('raw_runtime_backend', 'Unknown')}")
    print(f"Accelerator Runtime: {backend_info.get('is_gpu_build', False)}")
    print(f"Is Apple Silicon: {backend_info.get('is_apple_silicon', False)}")
    print(f"JAX Metal Active: {backend_info.get('jax_metal_active', False)}")
    print(f"Available Devices: {device_count}")

    if device_count > 0:
        print("\nDevice Information:")
        print("-" * 20)
        from .utils import get_device_info

        for i in range(device_count):
            device_info = get_device_info(i)
            print(f"\nDevice {i}:")
            print(f"  Name: {device_info.get('kind', 'Unknown')}")
            stats = device_info.get("memory_stats", {})
            if not device_info.get("memory_stats_available", False):
                print("  Device Memory: unavailable")
                print(
                    "  Reason: "
                    f"{device_info.get('memory_stats_error', 'not exposed by this backend')}"
                )
                continue

            allocated_bytes = stats.get("bytes_in_use", 0)
            reserved_bytes = stats.get("bytes_reserved")
            peak_bytes = stats.get("peak_bytes_in_use", 0)
            limit_bytes = stats.get("bytes_limit", 0)

            print(f"  Allocated Memory: {format_memory(allocated_bytes)}")
            if reserved_bytes is not None:
                print(f"  Reserved Memory: {format_memory(reserved_bytes)}")
            print(f"  Peak Memory: {format_memory(peak_bytes)}")
            if limit_bytes:
                print(f"  Device Limit: {format_memory(limit_bytes)}")

    return 0


def cmd_monitor(args: argparse.Namespace) -> int:
    """Monitor device memory usage in real-time."""
    if not JAX_AVAILABLE:
        print("Error: JAX not available")
        return 1

    print("Starting JAX memory monitoring...")
    print(f"Sampling interval: {args.interval} seconds")
    print(
        f"Duration: {args.duration} seconds"
        if args.duration is not None
        else "Duration: Indefinite"
    )
    if args.threshold:
        print(f"Alert threshold: {args.threshold} MB")
    print("Press Ctrl+C to stop\n")

    max_history = int(getattr(args, "max_history", 10000))
    tracker_class = _load_memory_tracker()
    tracker = tracker_class(
        sampling_interval=args.interval,
        alert_threshold_mb=args.threshold,
        device_index=args.device,
        enable_logging=True,
        max_history=max_history,
    )

    try:
        tracker.start_tracking()

        start_time = time.time()
        while True:
            if (
                args.duration is not None
                and (time.time() - start_time) >= args.duration
            ):
                break

            current_memory = tracker.get_current_memory()
            print(
                f"\rCurrent memory usage: {current_memory:.1f} MB", end="", flush=True
            )

            sleep_for = args.interval
            if args.duration is not None:
                remaining = args.duration - (time.time() - start_time)
                sleep_for = min(sleep_for, max(remaining, 0.0))
            time.sleep(sleep_for)

    except KeyboardInterrupt:
        print("\n\nStopping monitoring...")

    finally:
        results = tracker.stop_tracking()

        _report_monitor_results(results, args.output)

    return 0


def cmd_track(args: argparse.Namespace) -> int:
    """Start background memory tracking."""
    if not JAX_AVAILABLE:
        print("Error: JAX not available")
        return 1
    wandb_config = _resolve_wandb_config(args)
    if wandb_config is None:
        return 1
    mlflow_config = _resolve_mlflow_config(args)
    if mlflow_config is None:
        return 1

    tracker = _build_tracking_runtime(args)

    try:
        tracker.start_tracking()
        print("Tracking started. Press Ctrl+C to stop and save results.")

        with tracker.capture_oom(
            context="jaxmemprof.track",
            metadata={"command": "track", "runtime_backend": "jax"},
        ):
            while True:
                time.sleep(5.0)

                _print_tracking_status(tracker)

    except KeyboardInterrupt:
        print("\nStopping tracking...")

    finally:
        results = tracker.stop_tracking()

        _save_tracking_results(args, results)
        final_stats = _print_tracking_results(tracker, results)
        _export_tracking_integrations(
            args, tracker, results, final_stats, wandb_config, mlflow_config
        )

    return 0


def _build_tracking_runtime(args: argparse.Namespace) -> MemoryTracker:
    print("Starting background memory tracking...")
    job_id = getattr(args, "job_id", None)
    rank = getattr(args, "rank", None)
    local_rank = getattr(args, "local_rank", None)
    world_size = getattr(args, "world_size", None)
    telemetry_sink_config = _build_telemetry_sink_config(args)
    oom_flight_recorder = bool(getattr(args, "oom_flight_recorder", False))
    oom_dump_dir = str(getattr(args, "oom_dump_dir", "oom_dumps"))
    oom_buffer_size = getattr(args, "oom_buffer_size", None)
    oom_max_dumps = int(getattr(args, "oom_max_dumps", 5))
    oom_max_total_mb = int(getattr(args, "oom_max_total_mb", 256))
    max_history = int(getattr(args, "max_history", 10000))

    tracker_class = _load_memory_tracker()
    tracker = tracker_class(
        sampling_interval=args.interval,
        alert_threshold_mb=args.threshold,
        device_index=args.device,
        enable_logging=True,
        job_id=job_id,
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        telemetry_sink_config=telemetry_sink_config,
        save_device_profile_on_stop=args.profile,
        max_history=max_history,
        enable_oom_flight_recorder=oom_flight_recorder,
        oom_dump_dir=oom_dump_dir,
        oom_buffer_size=oom_buffer_size,
        oom_max_dumps=oom_max_dumps,
        oom_max_total_mb=oom_max_total_mb,
    )
    if telemetry_sink_config is not None:
        print(f"Append-only telemetry sink: {telemetry_sink_config.root_dir}")
    if oom_flight_recorder:
        print("OOM flight recorder enabled:")
        print(f"  Dump directory: {oom_dump_dir}")
        print(f"  Buffer size: {tracker.oom_buffer_size} events")
        print(f"  Max dumps: {oom_max_dumps}")
        print(f"  Max total size: {oom_max_total_mb} MB")

    def alert_callback(alert: Dict[str, Any]) -> None:
        print(f"\n⚠️  MEMORY ALERT: {alert['message']}")

    tracker.add_alert_callback(alert_callback)
    return tracker


def _print_tracking_status(tracker: MemoryTracker) -> None:
    stats = tracker.get_statistics()
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


def _save_tracking_results(args: argparse.Namespace, results: Any) -> None:
    if args.output:
        sampling_interval_ms = int(round(args.interval * 1000))
        capability = _tracking_memory_capability(results)
        output_data = {
            "peak_memory": results.peak_memory_bytes / (1024 * 1024),
            "average_memory": results.average_memory_bytes / (1024 * 1024),
            "duration": results.duration,
            "memory_usage": results.memory_usage,
            "timestamps": results.timestamps,
            "alerts": results.alert_count,
            "events": _normalize_telemetry_events(
                results.telemetry_events,
                sampling_interval_ms=sampling_interval_ms,
            ),
            "history_window_limit": getattr(results, "history_window_limit", 0),
            "history_retained_samples": getattr(results, "history_retained_samples", 0),
            "history_dropped_samples": getattr(results, "history_dropped_samples", 0),
            "history_retained_events": getattr(results, "history_retained_events", 0),
            "history_dropped_events": getattr(results, "history_dropped_events", 0),
            "history_retained_alerts": getattr(results, "history_retained_alerts", 0),
            "history_dropped_alerts": getattr(results, "history_dropped_alerts", 0),
            **capability,
        }

        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2)

        print(f"Results saved to {args.output}")


def _print_tracking_results(tracker: MemoryTracker, results: Any) -> Dict[str, Any]:
    final_stats = tracker.get_statistics()
    if _tracking_memory_capability(results)["device_memory_available"]:
        print(
            f"\nTracking completed. Peak memory: {results.peak_memory_bytes / (1024 * 1024):.1f} MB"
        )
    else:
        print("\nTracking completed. Device memory was unavailable on this backend.")
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

    if getattr(results, "device_memory_profile_path", None):
        print(f"Device memory profile saved to: {results.device_memory_profile_path}")
    if tracker.last_oom_dump_path:
        print(f"OOM flight recorder dump saved to: {tracker.last_oom_dump_path}")
    return final_stats


def _export_tracking_integrations(
    args: argparse.Namespace,
    tracker: MemoryTracker,
    results: Any,
    final_stats: Dict[str, Any],
    wandb_config: Any,
    mlflow_config: Any,
) -> None:
    if WANDB_AVAILABLE and wandb_config.enabled:
        try:
            export_tracking_run_to_wandb(
                wandb_config,
                command_name="jaxmemprof-track",
                session_summary=tracker.get_session_summary(),
                stats=final_stats,
                events=results.telemetry_events,
                output_path=args.output,
                telemetry_sink_dir=getattr(args, "telemetry_sink_dir", None),
                oom_dump_path=tracker.last_oom_dump_path,
            )
            print("W&B export completed.")
        except Exception as exc:
            _warn_wandb_export_failure("jaxmemprof track", exc)
    elif wandb_config.enabled:
        print("Warning: W&B is not available.", file=sys.stderr)

    if MLFLOW_AVAILABLE and mlflow_config.enabled:
        try:
            export_tracking_run_to_mlflow(
                mlflow_config,
                command_name="jaxmemprof-track",
                session_summary=tracker.get_session_summary(),
                stats=final_stats,
                events=results.telemetry_events,
                output_path=args.output,
                telemetry_sink_dir=getattr(args, "telemetry_sink_dir", None),
                oom_dump_path=tracker.last_oom_dump_path,
            )
            print("MLflow export completed.")
        except Exception as exc:
            _warn_mlflow_export_failure("jaxmemprof track", exc)
    elif mlflow_config.enabled:
        print("Warning: MLflow is not available.", file=sys.stderr)


def cmd_diagnose(args: argparse.Namespace) -> int:
    """Produce a portable diagnostic bundle. Returns 0 (OK), 1 (failure), or 2 (memory risk)."""
    if not JAX_AVAILABLE:
        print("Error: JAX not available")
        return 1
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
    run_diagnose = _load_run_diagnose()
    try:
        artifact_dir, exit_code = run_diagnose(
            output=args.output,
            device_index=args.device,
            duration=args.duration,
            interval=args.interval,
            command_line=command_line,
        )
    except OSError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    _print_diagnose_summary(artifact_dir, exit_code)
    _export_diagnose_integrations(artifact_dir, wandb_config, mlflow_config)

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
            with open(manifest_path, encoding="utf-8") as f:
                manifest = json.load(f)
            if manifest.get("risk_detected"):
                summary_path = artifact_dir / "diagnostic_summary.json"
                if summary_path.exists():
                    with open(summary_path, encoding="utf-8") as f:
                        summary = json.load(f)
                    flags = summary.get("risk_flags", {})
                    parts = [k for k, v in flags.items() if v]
                    if parts:
                        print(f"Findings: {', '.join(parts)}")
        if exit_code == 0 and status == "OK":
            print("Findings: no memory risk detected")
    except (OSError, json.JSONDecodeError):
        pass


def _export_diagnose_integrations(
    artifact_dir: Path, wandb_config: Any, mlflow_config: Any
) -> None:
    if WANDB_AVAILABLE and wandb_config.enabled:
        try:
            export_diagnose_bundle_to_wandb(
                wandb_config,
                command_name="jaxmemprof-diagnose",
                artifact_dir=artifact_dir,
            )
            print("W&B export completed.")
        except Exception as exc:
            _warn_wandb_export_failure("jaxmemprof diagnose", exc)
    elif wandb_config.enabled:
        print("Warning: W&B is not available.", file=sys.stderr)

    if MLFLOW_AVAILABLE and mlflow_config.enabled:
        try:
            export_diagnose_bundle_to_mlflow(
                mlflow_config,
                command_name="jaxmemprof-diagnose",
                artifact_dir=artifact_dir,
            )
            print("MLflow export completed.")
        except Exception as exc:
            _warn_mlflow_export_failure("jaxmemprof diagnose", exc)
    elif mlflow_config.enabled:
        print("Warning: MLflow is not available.", file=sys.stderr)


def cmd_analyze(args: argparse.Namespace) -> int:
    """Analyze a saved tracking result JSON file."""
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file {args.input} not found", file=sys.stderr)
        return 1

    try:
        with input_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error: Failed to load results from {args.input}: {e}", file=sys.stderr)
        return 1

    print(f"Analyzing JAX tracking results: {args.input}")
    print("=" * 50)
    memory_available = data.get("device_memory_available", True)
    if memory_available:
        print(f"Peak Memory: {data.get('peak_memory', 0.0):.2f} MB")
        print(f"Average Memory: {data.get('average_memory', 0.0):.2f} MB")
    else:
        print("Device Memory: unavailable")
        print(
            f"Reason: {data.get('device_memory_unavailable_reason', 'not exposed by backend')}"
        )
    print(f"Duration: {data.get('duration', 0.0):.2f} seconds")
    print(f"Alerts Triggered: {data.get('alerts', 0)}")

    class ResultWrapper:
        def __init__(self, d: Dict[str, Any]) -> None:
            self.memory_usage = d.get("memory_usage", [])
            self.timestamps = d.get("timestamps", [])
            self.peak_memory_mb = d.get("peak_memory", 0.0)
            self.average_memory_mb = d.get("average_memory", 0.0)
            self.snapshots: list[Any] = []

        @property
        def memory_growth_rate(self) -> float:
            if len(self.memory_usage) < 2 or not data.get("duration"):
                return 0.0
            memory_growth_mb = (
                float(self.memory_usage[-1]) - float(self.memory_usage[0])
            ) / (1024.0 * 1024.0)
            return memory_growth_mb / float(data["duration"])

    wrapper = ResultWrapper(data)

    _plot_analysis_timeline(args, wrapper, memory_available)

    if getattr(args, "detect_leaks", False) and memory_available:
        from .analyzer import MemoryAnalyzer

        findings = MemoryAnalyzer().detect_memory_leaks(wrapper)
        print(f"Potential memory leaks: {len(findings)}")
    if getattr(args, "optimize", False) and memory_available:
        from .analyzer import MemoryAnalyzer

        score = MemoryAnalyzer().score_optimization(wrapper)
        print(f"Optimization score: {score['overall_score']:.1f}/10")
    if getattr(args, "report", None):
        report = [
            "JAX Stormlog Analysis",
            f"Duration: {data.get('duration', 0):.2f} seconds",
        ]
        report.append("Device memory available: " + str(memory_available))
        Path(args.report).write_text("\n".join(report) + "\n", encoding="utf-8")
        print(f"Analysis report saved to {args.report}")

    return 0


def main() -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="JAX Stormlog CLI",
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
    subparsers.add_parser("info", help="Display system and device information")

    # Monitor command
    monitor_parser = subparsers.add_parser(
        "monitor", help="Monitor device memory usage"
    )
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
        default="0",
        help="JAX device index or backend: cpu, gpu, tpu, metal (default: 0)",
    )
    monitor_parser.add_argument("--output", help="Output file for results")
    monitor_parser.add_argument(
        "--max-history",
        type=int,
        default=10000,
        help="Maximum number of historical samples to keep in memory (default: 10000)",
    )

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
        default="0",
        help="JAX device index or backend: cpu, gpu, tpu, metal (default: 0)",
    )
    track_parser.add_argument(
        "--profile",
        action="store_true",
        help="Export a JAX device memory profile upon tracking completion",
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
    track_parser.add_argument(
        "--max-history",
        type=int,
        default=10000,
        help="Maximum number of historical samples to keep in memory (default: 10000)",
    )
    track_parser.add_argument(
        "--oom-flight-recorder",
        action="store_true",
        help="Enable automatic OOM flight recorder dump artifacts",
    )
    track_parser.add_argument(
        "--oom-dump-dir",
        default="oom_dumps",
        help="Directory used to write OOM dump bundles (default: oom_dumps)",
    )
    track_parser.add_argument(
        "--oom-buffer-size",
        type=int,
        default=None,
        help="Ring buffer size for OOM event dumps (default: max history)",
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
    add_wandb_arguments(track_parser)
    add_mlflow_arguments(track_parser)

    # Diagnose command
    diagnose_parser = subparsers.add_parser(
        "diagnose", help="Diagnose OOM dumps and memory issues"
    )
    diagnose_parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for the artifact bundle (default: cwd)",
    )
    diagnose_parser.add_argument(
        "--device",
        default="0",
        help="JAX device index or backend: cpu, gpu, tpu, metal (default: 0)",
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

    # Analyze command
    analyze_parser = subparsers.add_parser(
        "analyze", help="Analyze saved tracking results"
    )
    analyze_parser.add_argument(
        "--input", required=True, help="Input JSON file with tracking results"
    )
    analyze_parser.add_argument("--detect-leaks", action="store_true")
    analyze_parser.add_argument("--optimize", action="store_true")
    analyze_parser.add_argument("--visualize", action="store_true")
    analyze_parser.add_argument("--report", help="Write a text analysis report")
    analyze_parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for analysis results",
    )
    analyze_parser.add_argument(
        "--plot",
        nargs="?",
        const="memory_timeline.png",
        help="Generate a memory usage plot (default: memory_timeline.png)",
    )

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
    elif args.command == "diagnose":
        return cmd_diagnose(args)
    elif args.command == "analyze":
        return cmd_analyze(args)
    else:
        print(f"Unknown command: {args.command}")
        return 1


def _report_monitor_results(results: Any, output: str | None) -> None:
    print("\nMonitoring Results:")
    print("-" * 20)
    capability = _tracking_memory_capability(results)
    device_memory_available = capability["device_memory_available"]
    if device_memory_available:
        print(f"Peak Memory: {results.peak_memory_bytes / (1024 * 1024):.1f} MB")
        print(f"Average Memory: {results.average_memory_bytes / (1024 * 1024):.1f} MB")
    else:
        print("Device Memory: unavailable")
        print(
            f"Process RSS: {(capability['process_memory_bytes'] or 0) / (1024 * 1024):.1f} MB"
        )
    print(f"Duration: {results.duration:.1f} seconds")
    print(f"Samples Collected: {len(results.memory_usage)}")
    dropped_samples = getattr(results, "history_dropped_samples", 0)
    if dropped_samples:
        print(f"Dropped Samples: {dropped_samples}")

    if results.alert_count:
        print(f"Alerts Triggered: {results.alert_count}")

    if output:
        # Save results
        output_data = {
            "peak_memory": results.peak_memory_bytes / (1024 * 1024),
            "average_memory": results.average_memory_bytes / (1024 * 1024),
            "duration": results.duration,
            "memory_usage": results.memory_usage,
            "timestamps": results.timestamps,
            "alerts": results.alert_count,
            "history_window_limit": getattr(results, "history_window_limit", 0),
            "history_retained_samples": getattr(results, "history_retained_samples", 0),
            "history_dropped_samples": getattr(results, "history_dropped_samples", 0),
            **capability,
        }

        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2)

        print(f"Results saved to {output}")


def _plot_analysis_timeline(
    args: argparse.Namespace, wrapper: Any, memory_available: bool
) -> None:
    if (
        getattr(args, "plot", None) or getattr(args, "visualize", False)
    ) and memory_available:
        from .visualizer import MemoryVisualizer

        visualizer = MemoryVisualizer()

        plot = getattr(args, "plot", None)
        plot_name = plot if isinstance(plot, str) else "memory_timeline.png"

        if getattr(args, "output", None):
            output_dir = Path(args.output)
            output_dir.mkdir(parents=True, exist_ok=True)
            plot_path = str(output_dir / plot_name)
        else:
            plot_path = plot_name

        visualizer.plot_memory_timeline(wrapper, save_path=plot_path)
        print(f"Memory timeline plot saved to {plot_path}")


if __name__ == "__main__":
    sys.exit(main())
