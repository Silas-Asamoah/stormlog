"""Diagnostic bundle builder for the TensorFlow Stormlog diagnose command."""

import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from stormlog.derived_fields import compute_event_fields
from stormlog.session import (
    SESSION_STATUS_COMPLETED,
    SESSION_STATUS_INCOMPLETE,
    SessionSummary,
    create_session_summary,
    now_ns,
    session_summary_to_dict,
    update_session_summary,
)

from .utils import get_backend_info, get_gpu_info, get_system_info

# Risk thresholds (same semantics as Stormlog)
HIGH_UTILIZATION_RATIO = 0.85
MANIFEST_VERSION = 2


def _default_str(obj: Any) -> str:
    """JSON serializer for non-JSON-serializable types."""
    if hasattr(obj, "item"):  # numpy scalar
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _device_index(device: Optional[str]) -> int:
    """Parse device string (e.g. /GPU:0) to GPU index."""
    if not device or "/GPU:" not in device:
        return 0
    try:
        return int(device.split(":")[-1])
    except (ValueError, IndexError):
        return 0


def _create_artifact_dir(output: Optional[str], prefix: str) -> Path:
    """Create a collision-safe artifact directory."""
    ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")

    if output:
        out_path = Path(output).resolve()
        if out_path.exists() and out_path.is_dir():
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
    manifest = {
        "schema_version": MANIFEST_VERSION,
        "version": MANIFEST_VERSION,
        "created_iso": datetime.utcnow().isoformat() + "Z",
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
        json.dump(manifest, f, indent=2)


def collect_environment(device: Optional[str] = None) -> Dict[str, Any]:
    """Collect system and GPU data for the diagnostic bundle."""
    env: Dict[str, Any] = {}
    env["system"] = get_system_info()
    env["gpu"] = get_gpu_info()
    # TensorFlow does not expose fragmentation like PyTorch; omit or empty
    env["fragmentation"] = {
        "note": "TensorFlow does not expose fragmentation in this build"
    }
    return env


def run_timeline_capture(
    device: Optional[str],
    duration_seconds: float,
    interval: float,
) -> Dict[str, List[float]]:
    """
    Run tracker for the given duration and return timeline data.
    Returns empty timeline if duration_seconds <= 0 or on error.
    """
    if duration_seconds <= 0:
        return {"timestamps": [], "allocated": [], "reserved": []}

    try:
        from .tracker import MemoryTracker

        tracker = MemoryTracker(
            device=device or "/GPU:0",
            sampling_interval=interval,
            alert_threshold_mb=None,
            enable_logging=False,
        )
        tracker.start_tracking()
        try:
            time.sleep(duration_seconds)
        finally:
            result = tracker.stop_tracking()

        # Build timeline in the shared Stormlog shape: timestamps, allocated, reserved (bytes)
        timestamps = list(result.timestamps)
        # memory_usage is in MB; convert to bytes for allocated and reserved (TF uses same value)
        mb_to_bytes = 1024 * 1024
        allocated = [float(m) * mb_to_bytes for m in result.memory_usage]
        reserved = allocated.copy()
        return {
            "timestamps": timestamps,
            "allocated": allocated,
            "reserved": reserved,
        }
    except Exception:
        return {"timestamps": [], "allocated": [], "reserved": []}


def _suggest_tf_optimizations(utilization_ratio: float) -> List[str]:
    """Return TensorFlow-oriented suggestions for diagnostic summary."""
    suggestions: List[str] = []
    if utilization_ratio >= 0.9:
        suggestions.append(
            "Very high GPU memory utilization. Consider reducing batch size, "
            "using gradient checkpointing (tf.recompute_grad), or model parallelism."
        )
    if utilization_ratio >= HIGH_UTILIZATION_RATIO:
        suggestions.append(
            "Enable mixed precision with tf.keras.mixed_precision.Policy('mixed_float16') "
            "to reduce memory footprint."
        )
    suggestions.extend(
        [
            "Use tf.data for efficient input pipelines and memory use.",
            "Profile memory at different points in training to find bottlenecks.",
            "Consider clearing the session or limiting GPU growth with tf.config.experimental.set_memory_growth.",
        ]
    )
    return suggestions


def build_diagnostic_summary(
    device: Optional[str] = None,
) -> Tuple[Dict[str, Any], bool]:
    """
    Build diagnostic summary and risk flags from current state.
    Returns (summary_dict, risk_detected).
    """
    _system_info = get_system_info()
    backend_info = get_backend_info()
    backend = backend_info.get("runtime_backend", "cpu")
    gpu_info = get_gpu_info()

    # TensorFlow does not expose num_ooms or fragmentation; use 0
    num_ooms = 0
    fragmentation_ratio = 0.0

    allocated, peak, total_bytes = _device_memory_bytes(gpu_info, device)

    # TensorFlow's Python API (get_gpu_info) reports current_memory_mb and
    # peak_memory_mb — both track *allocated* memory. There is no separate
    # "reserved but unallocated" counter like PyTorch's
    # torch.cuda.memory_reserved(). We intentionally alias reserved_bytes to
    # allocated_bytes for schema compatibility across backend diagnostic
    # summaries. Downstream consumers MUST NOT interpret reserved_bytes from
    # a TF diagnostic summary as a real reservation signal.
    #
    # compute_event_fields gives us utilization_ratio and allocator_gap_bytes;
    # allocator_gap_bytes is always 0 for TF (reserved == allocated).
    # fragmentation_ratio remains 0.0 here (TF does not expose fragmentation).
    _synthetic_event = {
        "allocator_allocated_bytes": allocated,
        "allocator_reserved_bytes": allocated,
        "device_total_bytes": total_bytes if total_bytes else None,
        "collector": None,
    }
    _derived = compute_event_fields(_synthetic_event)
    utilization_ratio = _derived["utilization_ratio"] or 0.0
    allocator_gap_bytes = _derived["allocator_gap_bytes"] or 0  # always 0 for TF

    # Risk flags (no OOM/fragmentation from TF API)
    oom_occurred = num_ooms > 0
    high_utilization = total_bytes > 0 and utilization_ratio >= HIGH_UTILIZATION_RATIO
    fragmentation_warning = False  # TF does not expose
    risk_detected = oom_occurred or high_utilization or fragmentation_warning

    suggestions = _suggest_tf_optimizations(utilization_ratio)

    summary: Dict[str, Any] = {
        "backend": backend,
        "allocated_bytes": allocated,
        "reserved_bytes": allocated,  # TF has no separate reserved counter; intentional alias for schema compatibility
        "peak_bytes": peak,
        "total_bytes": total_bytes,
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
    return summary, risk_detected


def _device_memory_bytes(
    gpu_info: Dict[str, Any], device: Optional[str]
) -> Tuple[int, int, int]:
    # Per-device memory (TF reports in MB)
    devices = gpu_info.get("devices") or []
    idx = _device_index(device) if device else 0
    if devices and idx < len(devices):
        d = devices[idx]
        current_mb = d.get("current_memory_mb", 0) or 0
        peak_mb = d.get("peak_memory_mb", 0) or 0
        allocated = int(current_mb * 1024 * 1024)
        peak = int(peak_mb * 1024 * 1024)
        total_memory_mb = d.get("total_memory_mb")
        if isinstance(total_memory_mb, (int, float)) and total_memory_mb > 0:
            total_bytes = int(total_memory_mb * 1024 * 1024)
        else:
            total_bytes = 0
    else:
        allocated = 0
        peak = 0
        total_bytes = 0

    return allocated, peak, total_bytes


def run_diagnose(
    output: Optional[str],
    device: Optional[str],
    duration: float,
    interval: float,
    command_line: str,
) -> Tuple[Path, int]:
    """
    Build the full diagnostic bundle and write all artifact files.
    Returns (artifact_dir, exit_code).
    exit_code: 0 = success no risk, 1 = failure, 2 = success with memory risk.
    """
    try:
        artifact_dir = _create_artifact_dir(output, "stormlog-tensorflow-diagnose")
    except OSError as e:
        target = Path(output).resolve() if output else Path.cwd().resolve()
        print(f"Error: Cannot create output directory {target}: {e}", file=sys.stderr)
        raise

    session_summary = create_session_summary(
        source="tfmemprof diagnose",
        started_at_ns=now_ns(),
    )
    files_written: List[str] = []
    risk_detected = False
    exit_code = 0

    try:
        # 1. Environment
        env = collect_environment(device)
        env_path = artifact_dir / "environment.json"
        with open(env_path, "w") as f:
            json.dump(env, f, indent=2, default=_default_str)
        files_written.append("environment.json")

        # 2. Timeline (optional)
        timeline = run_timeline_capture(device, duration, interval)
        timeline_path = artifact_dir / "telemetry_timeline.json"
        with open(timeline_path, "w") as f:
            json.dump(timeline, f, indent=2, default=_default_str)
        files_written.append("telemetry_timeline.json")

        # 3. Diagnostic summary and risk
        summary, risk_detected = build_diagnostic_summary(device)
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
        files_written.append("manifest.json")
        _write_manifest(
            artifact_dir,
            command_line=command_line,
            files_written=files_written,
            exit_code=exit_code,
            risk_detected=risk_detected,
            session_summary=session_summary,
        )
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
