"""Diagnostic bundle builder for the JAX Stormlog diagnose command."""

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from stormlog.derived_fields import compute_event_fields
from stormlog.session import (
    SESSION_STATUS_COMPLETED,
    SESSION_STATUS_INCOMPLETE,
    SESSION_STATUS_RUNNING,
    SessionSummary,
    create_session_summary,
    now_ns,
    session_summary_to_dict,
    update_session_summary,
)

from .tracker import MemoryTracker
from .utils import get_backend_info, get_device_info, get_system_info

HIGH_UTILIZATION_RATIO = 0.85
MANIFEST_VERSION = 2


def _default_str(obj: Any) -> str:
    """JSON serializer for non-JSON-serializable types."""
    if hasattr(obj, "item"):  # numpy scalar
        return str(obj.item())
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _create_artifact_dir(output: Optional[str], prefix: str) -> Path:
    """Create a collision-safe artifact directory."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")

    if output:
        out_path = Path(output).resolve()
        if out_path.exists():
            if not out_path.is_dir():
                raise ValueError(
                    f"Output path exists but is not a directory: {out_path}"
                )
            base_dir = out_path
        else:
            out_path.mkdir(parents=True, exist_ok=False)
            return out_path
    else:
        base_dir = Path.cwd().resolve()

    base_name = f"{prefix}-{ts}"
    suffix = 0
    while True:
        name = base_name if suffix == 0 else f"{base_name}-{suffix}"
        artifact_dir = base_dir / name
        try:
            artifact_dir.mkdir(parents=True, exist_ok=False)
            return artifact_dir
        except FileExistsError:
            suffix += 1


def _write_manifest(
    artifact_dir: Path,
    *,
    command_line: str,
    files_written: list[str],
    exit_code: int,
    risk_detected: bool,
    session_summary: SessionSummary,
    error: str | None = None,
) -> None:
    manifest: Dict[str, Any] = {
        "schema_version": MANIFEST_VERSION,
        "version": MANIFEST_VERSION,
        "created_iso": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "command_line": command_line,
        "files": files_written,
        "exit_code": exit_code,
        "risk_detected": risk_detected,
        "session_id": session_summary.session_id,
        "session_status": session_summary.status,
        "session": session_summary_to_dict(session_summary),
    }
    if error:
        manifest["error"] = error
    manifest_path = artifact_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, default=_default_str)


def collect_environment(device_index: Union[int, str] = 0) -> Dict[str, Any]:
    """Collect system, JAX backend, and device environment details."""
    env: Dict[str, Any] = {}
    env["system"] = get_system_info()
    env["backend"] = get_backend_info()
    env["device"] = get_device_info(device_index)
    # JAX does not expose fragmentation like PyTorch; omit or empty
    env["fragmentation"] = {"note": "JAX does not expose fragmentation in this build"}
    return env


def run_timeline_capture(
    device_index: Union[int, str], duration_seconds: float, interval: float
) -> Dict[str, Any]:
    """Capture a timeline of memory metrics by running the tracker briefly.

    Returns timeline data in the shared Stormlog shape: timestamps,
    allocated (bytes), reserved (bytes).
    """
    if duration_seconds <= 0:
        return {
            "timestamps": [],
            "allocated": [],
            "reserved": [],
            "device_memory_available": False,
        }

    try:
        tracker = MemoryTracker(
            sampling_interval=interval,
            device_index=device_index,
            enable_logging=False,
        )
        tracker.start_tracking()
        try:
            time.sleep(duration_seconds)
        finally:
            result = tracker.stop_tracking()

        memory_available = bool(getattr(result, "device_memory_available", True))
        if not memory_available:
            return {
                "timestamps": [],
                "allocated": [],
                "reserved": [],
                "device_memory_available": False,
                "device_memory_unavailable_reason": getattr(
                    result, "device_memory_unavailable_reason", None
                ),
                "process_memory_bytes": getattr(result, "process_memory_bytes", None),
            }

        # Try to reconstruct timeline with actual reserved bytes from telemetry events
        timestamps = []
        allocated = []
        reserved = []
        for event in getattr(result, "telemetry_events", []):
            if event.get("event_type") == "sample":
                timestamps.append(event.get("timestamp_ns", 0) / 1e9)
                allocated.append(float(event.get("allocator_allocated_bytes", 0)))
                reserved.append(float(event.get("allocator_reserved_bytes", 0)))

        if not timestamps:
            timestamps = list(result.timestamps)
            allocated = [float(m) for m in result.memory_usage]
            reserved = allocated.copy()

        return {
            "timestamps": timestamps,
            "allocated": allocated,
            "reserved": reserved,
            "device_memory_available": True,
        }
    except Exception:
        return {
            "timestamps": [],
            "allocated": [],
            "reserved": [],
            "device_memory_available": False,
        }


def _suggest_jax_optimizations(utilization_ratio: float) -> List[str]:
    """Provide basic optimizations for JAX memory based on telemetry."""
    suggestions: List[str] = []
    if utilization_ratio >= 0.9:
        suggestions.append(
            "Very high device utilization. Consider reducing batch size, "
            "using gradient checkpointing (jax.checkpoint), or model parallelism."
        )
    if utilization_ratio >= HIGH_UTILIZATION_RATIO:
        suggestions.append(
            "High device utilization detected. "
            "Consider using `jax.clear_caches()` between steps "
            "if memory is unexpectedly held."
        )
    suggestions.extend(
        [
            "Ensure XLA memory preallocation "
            "(`XLA_PYTHON_CLIENT_PREALLOCATE=true`) is tuned for your workload.",
            "Profile memory at different points in training to find bottlenecks.",
            "Consider using `XLA_PYTHON_CLIENT_MEM_FRACTION` to limit JAX "
            "device memory allocation.",
        ]
    )
    return suggestions


def build_diagnostic_summary(
    device_index: Union[int, str] = 0,
) -> Tuple[Dict[str, Any], bool]:
    """Build diagnostic summary and risk flags from current state.

    Returns (summary_dict, risk_detected). Summary schema matches
    the TensorFlow backend for downstream compatibility.
    """
    device_info = get_device_info(device_index)
    backend_info = get_backend_info()
    backend = backend_info.get("runtime_backend", "cpu")
    stats = device_info.get("memory_stats", {})

    allocated, peak, limit_bytes = _allocator_counters(stats)

    # Use actual reserved bytes from memory stats when available
    reserved_val = stats.get("bytes_reserved")
    is_approximate = False
    if reserved_val is not None:
        reserved = int(reserved_val)
    else:
        reserved = allocated
        is_approximate = True

    # compute_event_fields expects a mapping with allocator counter keys
    _synthetic_event = {
        "allocator_allocated_bytes": allocated,
        "allocator_reserved_bytes": reserved,
        "device_total_bytes": limit_bytes if limit_bytes else None,
        "collector": None,
    }
    _derived = compute_event_fields(_synthetic_event)
    utilization_ratio = _derived["utilization_ratio"] or 0.0
    allocator_gap_bytes = _derived["allocator_gap_bytes"] or 0

    # JAX does not expose OOM counts or fragmentation
    num_ooms = 0
    fragmentation_ratio = 0.0

    # Risk flags
    oom_occurred = num_ooms > 0
    high_utilization = limit_bytes > 0 and utilization_ratio >= HIGH_UTILIZATION_RATIO
    fragmentation_warning = False
    risk_detected = oom_occurred or high_utilization or fragmentation_warning

    suggestions = _suggest_jax_optimizations(utilization_ratio)

    summary: Dict[str, Any] = {
        "backend": backend,
        "allocated_bytes": allocated,
        "reserved_bytes": reserved,
        "peak_bytes": peak,
        "total_bytes": limit_bytes,
        "allocator_gap_bytes": allocator_gap_bytes,
        "utilization_ratio": utilization_ratio,
        "fragmentation_ratio": fragmentation_ratio,
        "num_ooms": num_ooms,
        "risk_flags": {
            "oom_occurred": oom_occurred,
            "high_utilization": high_utilization,
            "fragmentation_warning": fragmentation_warning,
        },
        "suggestions": suggestions,
    }
    if is_approximate:
        summary["allocator_reserved_approximate"] = True
    return summary, risk_detected


def run_diagnose(
    output: Optional[str],
    device_index: Union[int, str],
    duration: float,
    interval: float,
    command_line: str,
) -> Tuple[Path, int]:
    """Build the full diagnostic bundle and write all artifact files.

    Returns (artifact_dir, exit_code).
    exit_code: 0 = success no risk, 1 = failure, 2 = success with memory risk.
    """
    try:
        artifact_dir = _create_artifact_dir(output, "stormlog-jax-diagnose")
    except OSError as e:
        target = Path(output).resolve() if output else Path.cwd().resolve()
        print(f"Error: Cannot create output directory {target}: {e}", file=sys.stderr)
        raise

    session_summary = create_session_summary(
        source="stormlog.jax.diagnose",
        status=SESSION_STATUS_RUNNING,
        started_at_ns=now_ns(),
    )
    files_written: List[str] = []
    risk_detected = False
    exit_code = 0

    try:
        # 1. Environment
        env = collect_environment(device_index)
        env_path = artifact_dir / "environment.json"
        with open(env_path, "w") as f:
            json.dump(env, f, indent=2, default=_default_str)
        files_written.append("environment.json")

        # 2. Timeline
        timeline = run_timeline_capture(device_index, duration, interval)
        timeline_path = artifact_dir / "telemetry_timeline.json"
        with open(timeline_path, "w") as f:
            json.dump(timeline, f, indent=2, default=_default_str)
        files_written.append("telemetry_timeline.json")

        # 3. Diagnostic summary and risk
        summary, risk_detected = build_diagnostic_summary(device_index)
        summary_path = artifact_dir / "diagnostic_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=_default_str)
        files_written.append("diagnostic_summary.json")

        exit_code = 2 if risk_detected else 0

        # 4. Manifest
        session_summary = update_session_summary(
            session_summary,
            status=SESSION_STATUS_COMPLETED,
            ended_at_ns=now_ns(),
        )
        _write_manifest(
            artifact_dir,
            command_line=command_line,
            files_written=files_written + ["manifest.json"],
            exit_code=exit_code,
            risk_detected=risk_detected,
            session_summary=session_summary,
        )
        files_written.append("manifest.json")
    except OSError as e:
        print(f"Error: Failed to write diagnostic artifact: {e}", file=sys.stderr)
        exit_code = 1
        if not files_written:
            raise
        session_summary = update_session_summary(
            session_summary,
            status=SESSION_STATUS_INCOMPLETE,
            ended_at_ns=now_ns(),
        )
        try:
            files_with_manifest = list(files_written)
            if "manifest.json" not in files_with_manifest:
                files_with_manifest.append("manifest.json")
            _write_manifest(
                artifact_dir,
                command_line=command_line,
                files_written=files_with_manifest,
                exit_code=1,
                risk_detected=risk_detected,
                session_summary=session_summary,
                error=str(e),
            )
        except OSError:
            pass

    return artifact_dir, exit_code


def _allocator_counters(stats: Dict[str, Any]) -> Tuple[int, int, int]:
    allocated = int(stats.get("bytes_in_use", 0) or 0)
    peak = int(stats.get("peak_bytes_in_use", 0) or 0)
    limit_bytes = int(stats.get("bytes_limit", 0) or 0)

    return allocated, peak, limit_bytes
