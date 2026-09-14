"""Diagnostic bundle builder for the Stormlog diagnose command."""

import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .cuda_native_debug import (
    DEFAULT_TRACE_ALLOC_MAX_ENTRIES,
    capture_cuda_snapshot_artifacts,
    cuda_memory_history,
    cuda_memory_history_supported,
)
from .derived_fields import compute_event_fields
from .device_collectors import (
    build_device_memory_collector,
    detect_torch_runtime_backend,
)
from .session import (
    SESSION_STATUS_COMPLETED,
    SESSION_STATUS_INCOMPLETE,
    SessionSummary,
    create_session_summary,
    now_ns,
    session_summary_to_dict,
    update_session_summary,
)
from .utils import (
    check_memory_fragmentation,
    get_gpu_info,
    get_system_info,
    suggest_memory_optimization,
)

# Risk thresholds (align with suggest_memory_optimization where applicable)
HIGH_UTILIZATION_RATIO = 0.85
FRAGMENTATION_WARNING_RATIO = 0.3
MANIFEST_VERSION = 2


def _default_str(obj: Any) -> str:
    """JSON serializer for non-JSON-serializable types."""
    if hasattr(obj, "item"):  # numpy scalar
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _collect_backend_sample(
    device: Optional[int],
) -> Tuple[Optional[str], Optional[Any]]:
    """Collect one backend-aware sample when a GPU runtime is available."""
    runtime_backend = detect_torch_runtime_backend()
    if runtime_backend not in {"cuda", "rocm", "mps"}:
        return None, None
    try:
        collector_device: Optional[Any]
        if runtime_backend == "mps":
            collector_device = "mps"
        else:
            collector_device = device
        collector = build_device_memory_collector(collector_device)
        return runtime_backend, collector.sample()
    except Exception:
        return runtime_backend, None


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
    native_history: bool,
    error: str | None = None,
) -> None:
    manifest = {
        "schema_version": MANIFEST_VERSION,
        "version": MANIFEST_VERSION,
        "created_iso": datetime.utcnow().isoformat() + "Z",
        "command_line": command_line,
        "native_history_enabled": native_history,
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


def collect_environment(device: Optional[int] = None) -> Dict[str, Any]:
    """Collect system, GPU, and fragmentation data for the diagnostic bundle."""
    env: Dict[str, Any] = {}
    system_info = get_system_info()
    env["system"] = system_info
    runtime_backend = str(system_info.get("detected_backend", "cpu"))

    gpu_info = get_gpu_info(device)
    if runtime_backend == "mps":
        _, sample = _collect_backend_sample(device)
        if sample is not None:
            gpu_info = {
                "backend": "mps",
                "device_id": sample.device_id,
                "allocated_memory": sample.allocated_bytes,
                "reserved_memory": sample.reserved_bytes,
                "total_memory": sample.total_bytes,
            }
    env["gpu"] = gpu_info

    if runtime_backend in {"cuda", "rocm"} and not gpu_info.get("error"):
        frag = check_memory_fragmentation(device)
    elif runtime_backend == "mps":
        frag = {"note": "MPS fragmentation metrics are not available"}
    else:
        frag = {"error": "CUDA is not available"}
    env["fragmentation"] = frag

    return env


def run_timeline_capture(
    device: Optional[int],
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
        runtime_backend = detect_torch_runtime_backend()
        if runtime_backend in {"cuda", "rocm", "mps"}:
            from .tracker import MemoryTracker

            tracker_device = "mps" if runtime_backend == "mps" else device
            tracker: Any = MemoryTracker(
                device=tracker_device,
                sampling_interval=interval,
                enable_alerts=False,
            )
        else:
            from .cpu_profiler import CPUMemoryTracker

            tracker = CPUMemoryTracker(sampling_interval=interval)

        tracker.start_tracking()
        try:
            time.sleep(duration_seconds)
        finally:
            tracker.stop_tracking()

        timeline = tracker.get_memory_timeline(interval=interval)
        return {
            "timestamps": list(timeline.get("timestamps", [])),
            "allocated": list(timeline.get("allocated", [])),
            "reserved": list(timeline.get("reserved", [])),
        }
    except Exception:
        return {"timestamps": [], "allocated": [], "reserved": []}


def _diagnostic_memory_state(
    backend: str, gpu_info: Dict[str, Any], device: Optional[int]
) -> Tuple[int, int, int, int, Dict[str, Any]]:
    """Collect allocated, reserved, total, peak, and backend fragmentation state."""
    frag_info: Dict[str, Any] = {}

    # Current memory state
    allocated = 0
    reserved = 0
    total = 0
    peak = 0

    if backend in {"cuda", "rocm"} and not gpu_info.get("error"):
        frag_info = check_memory_fragmentation(device)
        allocated = gpu_info.get("allocated_memory", 0) or 0
        reserved = gpu_info.get("reserved_memory", 0) or 0
        total = gpu_info.get("total_memory") or 0
        peak = gpu_info.get("max_memory_allocated", 0) or allocated
    elif backend == "mps":
        _, sample = _collect_backend_sample(device)
        if sample is not None:
            allocated = sample.allocated_bytes
            reserved = sample.reserved_bytes
            total = sample.total_bytes or 0
            peak = max(allocated, reserved)

    return allocated, reserved, total, peak, frag_info


def _diagnostic_oom_count(backend: str, gpu_info: Dict[str, Any]) -> int:
    if (
        backend in {"cuda", "rocm"}
        and "memory_stats" in gpu_info
        and isinstance(gpu_info["memory_stats"], dict)
    ):
        return gpu_info["memory_stats"].get("num_ooms", 0) or 0
    return 0


def _diagnostic_suggestions(
    backend: str,
    gpu_info: Dict[str, Any],
    frag_info: Dict[str, Any],
    high_utilization: bool,
) -> List[str]:
    if backend in {"cuda", "rocm"} and not gpu_info.get("error"):
        return suggest_memory_optimization(frag_info)
    if backend == "mps" and high_utilization:
        return [
            "High MPS memory utilization detected. Consider reducing batch size or using mixed precision."
        ]
    return []


def build_diagnostic_summary(
    device: Optional[int] = None,
) -> Tuple[Dict[str, Any], bool]:
    """
    Build diagnostic summary and risk flags from current state.
    Returns (summary_dict, risk_detected).
    """
    system_info = get_system_info()
    backend = system_info.get("detected_backend", "cpu")
    gpu_info = get_gpu_info(device)
    allocated, reserved, total, peak, frag_info = _diagnostic_memory_state(
        backend, gpu_info, device
    )
    _synthetic_event = {
        "allocator_allocated_bytes": allocated,
        "allocator_reserved_bytes": reserved,
        "device_total_bytes": total if total else None,
        "collector": None,
    }
    _derived = compute_event_fields(_synthetic_event)
    utilization_ratio = _derived["utilization_ratio"] or 0.0
    allocator_gap_bytes = _derived["allocator_gap_bytes"] or 0
    # fragmentation_ratio comes from check_memory_fragmentation(), which uses
    # torch.cuda.memory_stats() — a richer source than the simple
    # (reserved - allocated) / reserved approximation in compute_event_fields.
    fragmentation_ratio = float(frag_info.get("fragmentation_ratio", 0))
    num_ooms = _diagnostic_oom_count(backend, gpu_info)

    # Risk flags
    oom_occurred = num_ooms > 0
    high_utilization = total > 0 and utilization_ratio >= HIGH_UTILIZATION_RATIO
    fragmentation_warning = fragmentation_ratio >= FRAGMENTATION_WARNING_RATIO
    risk_detected = oom_occurred or high_utilization or fragmentation_warning

    suggestions = _diagnostic_suggestions(
        backend, gpu_info, frag_info, high_utilization
    )

    summary: Dict[str, Any] = {
        "backend": backend,
        "allocated_bytes": allocated,
        "reserved_bytes": reserved,
        "peak_bytes": peak,
        "total_bytes": total,
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


def _validate_native_history_request() -> None:
    runtime_backend = detect_torch_runtime_backend()
    if runtime_backend != "cuda":
        raise RuntimeError(
            "Native memory history is currently supported only for CUDA runtimes."
        )
    if not cuda_memory_history_supported():
        raise RuntimeError(
            "Native CUDA memory history is unavailable in this PyTorch runtime."
        )


def run_diagnose(
    output: Optional[str],
    device: Optional[int],
    duration: float,
    interval: float,
    command_line: str,
    native_history: bool = False,
    native_history_max_entries: int = DEFAULT_TRACE_ALLOC_MAX_ENTRIES,
) -> Tuple[Path, int]:
    """
    Build the full diagnostic bundle and write all artifact files.
    Returns (artifact_dir, exit_code).
    exit_code: 0 = success no risk, 1 = failure, 2 = success with memory risk.
    """
    _validate_native_history_options(native_history, native_history_max_entries)

    try:
        artifact_dir = _create_artifact_dir(output, "stormlog-diagnose")
    except OSError as e:
        target = Path(output).resolve() if output else Path.cwd().resolve()
        print(f"Error: Cannot create output directory {target}: {e}", file=sys.stderr)
        raise

    session_summary = create_session_summary(
        source="gpumemprof diagnose",
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
        if native_history:
            with cuda_memory_history(
                device=device,
                trace_alloc_max_entries=native_history_max_entries,
            ):
                timeline = run_timeline_capture(device, duration, interval)
                files_written.extend(
                    capture_cuda_snapshot_artifacts(
                        artifact_dir,
                        device=device,
                        history_recorded=True,
                    )
                )
        else:
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

        # 4. Manifest (last, so it can include exit_code and risk_detected)
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
            native_history=native_history,
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
                native_history=native_history,
                error=str(e),
            )
        except OSError:
            pass

    return artifact_dir, exit_code


def _validate_native_history_options(
    native_history: bool, native_history_max_entries: int
) -> None:
    if native_history:
        if native_history_max_entries <= 0:
            raise ValueError("native_history_max_entries must be >= 1")
        _validate_native_history_request()
