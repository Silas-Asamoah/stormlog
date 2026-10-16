"""Offline MLX allocator analysis using the canonical session loader."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence, cast

from stormlog.session import select_default_loaded_session, session_summary_to_dict
from stormlog.telemetry import (
    LoadedTelemetrySession,
    TelemetryEvent,
    load_telemetry_sessions,
)

from .profile_artifact import FORMAT as PROFILE_FORMAT
from .profile_artifact import load_profiles

_MEASUREMENTS = {"start", "sample", "stop", "phase_start", "phase_end"}


def analyze(path: str | Path, *, session_id: str | None = None) -> dict[str, Any]:
    """Select a session by the shared policy; never discover a native runtime."""
    source = Path(path).resolve()
    if source.is_dir() and (source / "telemetry").is_dir():
        source = source / "telemetry"
    if source.is_file() and source.suffix == ".json":
        payload = json.loads(source.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and payload.get("format") == PROFILE_FORMAT:
            return _analyze_profiles(source, session_id)
    sessions = load_telemetry_sessions(source, permissive_legacy=False)
    selected = _select_session(sessions, session_id)
    return analyze_session(selected, source=str(source))


def _select_session(
    sessions: Sequence[LoadedTelemetrySession], session_id: str | None
) -> LoadedTelemetrySession:
    if session_id is not None:
        for session in sessions:
            if session.summary.session_id == session_id:
                return session
        raise ValueError(f"Requested session_id not found: {session_id}")
    selected = select_default_loaded_session(sessions)
    if selected is None:
        raise ValueError("No telemetry sessions found")
    return cast(LoadedTelemetrySession, selected)


def analyze_session(session: LoadedTelemetrySession, *, source: str) -> dict[str, Any]:
    events = sorted(session.events, key=lambda e: e.timestamp_ns)
    if not events or any(e.metadata.get("framework") != "mlx" for e in events):
        raise ValueError("Expected a nonempty MLX telemetry session")
    samples = [e for e in events if e.event_type in _MEASUREMENTS]
    valid = [e for e in samples if e.allocator_allocated_bytes is not None]
    windows = _growth_windows(samples)
    gaps = len(samples) - len(valid)
    return {
        "schema_version": 1,
        "format": "stormlog.mlx.analysis",
        "source": source,
        "session": session_summary_to_dict(session.summary),
        "memory_scope": "process_allocator",
        "metrics": _sample_metrics(samples, valid),
        "growth_windows": windows,
        "timeline": _timeline(samples),
        "findings": _session_findings(session, gaps, windows),
        "warnings": list(session.warnings),
        "notes": [
            "Sampled maxima can miss short-lived peaks; native peaks span process lifetime or external resets.",
            "Active growth is a candidate finding, not proof of a leak; warmup, compilation and retained outputs can explain it.",
            "Cache is reusable allocator storage, not fragmentation. RSS can overlap unified allocator pages.",
        ],
    }


def _sample_metrics(
    samples: Sequence[TelemetryEvent], valid: Sequence[TelemetryEvent]
) -> dict[str, Any]:
    return {
        "observed_samples": len(samples),
        "valid_samples": len(valid),
        "unavailable_samples": len(samples) - len(valid),
        "sampled_peak_bytes": _maximum([e.allocator_allocated_bytes for e in valid]),
        "cache_peak_bytes": _maximum([_mlx_bytes(e, "cache_bytes") for e in samples]),
        "runtime_lifetime_peak_bytes": _maximum(
            [_mlx_bytes(e, "runtime_peak_bytes") for e in samples]
        ),
        "device_total_bytes": None,
        "device_free_bytes": None,
        "utilization_ratio": None,
        "fragmentation_ratio": None,
    }


def _timeline(samples: Sequence[TelemetryEvent]) -> list[dict[str, Any]]:
    return [
        {
            "timestamp_ns": e.timestamp_ns,
            "active_bytes": e.allocator_allocated_bytes,
            "cache_bytes": _mlx_bytes(e, "cache_bytes"),
            "runtime_peak_bytes": _mlx_bytes(e, "runtime_peak_bytes"),
            "process_rss_bytes": e.metadata.get("process_rss_bytes"),
        }
        for e in samples
    ]


def _session_findings(
    session: LoadedTelemetrySession, gaps: int, windows: Sequence[dict[str, int]]
) -> list[dict[str, str]]:
    findings = []
    if gaps:
        findings.append(
            _finding(
                "collector_gaps",
                "Allocator collection has gaps",
                "/metrics/unavailable_samples",
            )
        )
    if any(w["growth_bytes"] > 0 for w in windows):
        findings.append(
            _finding(
                "active_growth",
                "Active allocator growth needs workload review",
                "/growth_windows",
            )
        )
    if session.summary.status != "completed":
        findings.append(
            _finding(
                "incomplete_capture", "Capture did not complete", "/session/status"
            )
        )
    return findings


def _finding(kind: str, title: str, pointer: str) -> dict[str, str]:
    return {"kind": kind, "title": title, "pointer": pointer}


def _mlx_bytes(event: TelemetryEvent, key: str) -> int | None:
    metadata = event.metadata.get("mlx")
    value = metadata.get(key) if isinstance(metadata, dict) else None
    return value if type(value) is int and value >= 0 else None


def _maximum(values: Sequence[int | None]) -> int | None:
    known = [v for v in values if v is not None]
    return max(known) if known else None


def _growth_windows(samples: Sequence[TelemetryEvent]) -> list[dict[str, int]]:
    windows: list[list[TelemetryEvent]] = []
    current: list[TelemetryEvent] = []
    for event in samples:
        if event.allocator_allocated_bytes is None:
            current = []
            continue
        if not current or _breaks_continuity(current[-1], event):
            current = []
            windows.append(current)
        current.append(event)
    return [_window_summary(window) for window in windows if len(window) > 1]


def _breaks_continuity(previous: TelemetryEvent, event: TelemetryEvent) -> bool:
    interval = max(previous.sampling_interval_ms, event.sampling_interval_ms)
    return event.allocator_change_bytes is None or (
        interval > 0
        and event.timestamp_ns - previous.timestamp_ns > interval * 3_000_000
    )


def _window_summary(window: Sequence[TelemetryEvent]) -> dict[str, int]:
    first, last = window[0], window[-1]
    assert first.allocator_allocated_bytes is not None
    assert last.allocator_allocated_bytes is not None
    return {
        "start_ns": first.timestamp_ns,
        "end_ns": last.timestamp_ns,
        "samples": len(window),
        "start_active_bytes": first.allocator_allocated_bytes,
        "end_active_bytes": last.allocator_allocated_bytes,
        "growth_bytes": last.allocator_allocated_bytes
        - first.allocator_allocated_bytes,
    }


def _analyze_profiles(source: Path, session_id: str | None) -> dict[str, Any]:
    profiles = load_profiles(source)["profiles"]
    if session_id is not None:
        profiles = [
            p for p in profiles if p["session_summary"]["session_id"] == session_id
        ]
        if not profiles:
            raise ValueError(f"Requested session_id not found: {session_id}")
    return {
        "schema_version": 1,
        "format": "stormlog.mlx.analysis",
        "source": str(source),
        "memory_scope": "process_allocator",
        "profiles": profiles,
        "metrics": {
            "profile_count": len(profiles),
            "sampled_peak_bytes": _maximum([p["sampled_peak_bytes"] for p in profiles]),
        },
        "findings": (
            [
                _finding(
                    "incomplete_compute",
                    "A profile has unverified completion",
                    "/profiles",
                )
            ]
            if any(
                not p["completion_verified"] or p["status"] != "completed"
                for p in profiles
            )
            else []
        ),
        "notes": [
            "Elapsed time is synchronized host time only when completion_verified is true."
        ],
    }
