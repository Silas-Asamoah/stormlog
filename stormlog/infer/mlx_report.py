"""Offline process-scoped MLX evidence, separate from client/engine statistics."""

from __future__ import annotations

import math
import urllib.parse
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .mlx_evidence import LoadedEvidence, load_all_evidence
from .omlx_stats import stats_report
from .server_clock import (
    ServerClock,
    artifact_alignments,
    build_server_clock,
    client_clock_domain,
)

MEMORY_FIELDS = {
    "active_allocator_bytes": (
        "allocator_allocated_bytes",
        "process_mlx_allocator_active",
    ),
    "cache_allocator_bytes": (
        "metadata.mlx.cache_bytes",
        "process_mlx_allocator_cache",
    ),
    "runtime_peak_bytes": (
        "metadata.mlx.runtime_peak_bytes",
        "process_since_start_or_external_reset",
    ),
    "process_rss_bytes": ("metadata.process_rss_bytes", "process_rss"),
}


def mlx_evidence_report(
    records: list[dict[str, Any]],
    paths: Sequence[str | Path],
    *,
    clock_offset_ns: int | None = None,
    clock_uncertainty_ns: int | None = None,
) -> dict[str, Any]:
    evidence = load_all_evidence(paths)
    artifact = _artifact(records)
    states = [_association(item, records, artifact) for item in evidence]
    _mark_ambiguity(evidence, states)
    processes = [
        _process(item, state, records, artifact, clock_offset_ns, clock_uncertainty_ns)
        for item, state in zip(evidence, states)
    ]
    return {
        "available": bool(evidence),
        "observation_scope": "sampled_server_processes",
        "processes": processes,
        "omlx_stats": stats_report(records),
        "aggregation": "none_process_memory_is_not_disjoint_device_memory",
        "request_execution_attribution": "unavailable",
        "scheduler_attribution": "unavailable",
        "prefill_decode_attribution": "unavailable",
        "kernel_attribution": "unavailable",
        "device_capacity": None,
        "reserved_allocator": None,
        "association_counts": {
            state: sum(
                process["association"]["state"] == state for process in processes
            )
            for state in ("verified", "declared_direct", "ambiguous", "unjoined")
        },
    }


def _artifact(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    artifacts = [
        record for record in records if record.get("event_type") == "infer.artifact"
    ]
    unique = {
        tuple(
            record.get("context", {}).get(key)
            for key in ("run_id", "session_id", "clock_domain", "host", "pid")
        )
        for record in artifacts
    }
    return artifacts[0] if artifacts and len(unique) == 1 else None


def _client_endpoints(
    records: list[dict[str, Any]],
) -> set[tuple[str, str, int | None]]:
    urls = [
        record.get("endpoint")
        for record in records
        if record.get("event_type") in {"infer.request", "infer.audio.request"}
    ]
    if not urls:
        urls = [
            record.get("config", {}).get("endpoint")
            for record in records
            if record.get("event_type") == "infer.session"
        ]
    return {_origin(url) for url in urls if isinstance(url, str)}


def _origin(url: str) -> tuple[str, str, int | None]:
    parsed = urllib.parse.urlparse(url)
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    return parsed.scheme, parsed.hostname or "", port


def _association(
    item: LoadedEvidence, records: list[dict[str, Any]], artifact: dict[str, Any] | None
) -> dict[str, Any]:
    state: dict[str, Any] = {
        "state": "unjoined",
        "reason": None,
        "run_id_match": False,
        "physical_process_epoch_known": item.manifest["process"]["start_time_ns"]
        is not None,
        "request_ownership": "unavailable",
    }
    if artifact is None:
        state["reason"] = "client_artifact_identity_missing_or_ambiguous"
        return state
    if artifact["context"]["run_id"] != item.run_id:
        state["reason"] = "run_id_mismatch"
        return state
    state["run_id_match"] = True
    endpoint = item.manifest["endpoint"]
    expected = ("http", endpoint["host"] or "", endpoint["port"])
    if expected not in _client_endpoints(records):
        state["reason"] = "endpoint_not_declared_or_different"
        return state
    state.update(
        state="declared_direct", reason="declared_endpoint_and_run_id_not_routing_proof"
    )
    if _verified_process_association(item, artifact):
        state.update(
            state="verified",
            reason="same_host_boot_and_observed_qualified_launcher_listener_pid",
        )
    return state


def _verified_process_association(
    item: LoadedEvidence, artifact: dict[str, Any]
) -> bool:
    proof = item.manifest["endpoint"]["routing_proof"]
    return bool(
        proof
        and item.manifest["producer"]["mode"] == "mlx-serve"
        and item.manifest["process"]["start_time_ns"] is not None
        and item.manifest["clock"]["boot_qualified"]
        and item.clock_domain == client_clock_domain(artifact)
    )


def _mark_ambiguity(
    evidence: list[LoadedEvidence], states: list[dict[str, Any]]
) -> None:
    for index, item in enumerate(evidence):
        if states[index]["state"] == "unjoined":
            continue
        for other_index in range(index):
            other = evidence[other_index]
            if states[other_index]["state"] == "unjoined":
                continue
            if _conflicting_endpoint_epochs(item, other):
                states[index].update(
                    state="ambiguous",
                    reason="overlapping_or_unplaced_process_epochs_for_endpoint",
                )
                states[other_index].update(
                    state="ambiguous",
                    reason="overlapping_or_unplaced_process_epochs_for_endpoint",
                )


def _conflicting_endpoint_epochs(left: LoadedEvidence, right: LoadedEvidence) -> bool:
    a, b = left.manifest["endpoint"], right.manifest["endpoint"]
    if (a["host"], a["port"]) != (b["host"], b["port"]):
        return False
    if left.clock_domain != right.clock_domain:
        return True
    a, b = left.manifest["capture"], right.manifest["capture"]
    return bool(
        max(a["started_at_ns"], b["started_at_ns"])
        <= min(a["ended_at_ns"], b["ended_at_ns"])
    )


def _clock(
    item: LoadedEvidence,
    records: list[dict[str, Any]],
    artifact: dict[str, Any] | None,
    offset: int | None,
    uncertainty: int | None,
) -> ServerClock | str:
    domain = client_clock_domain(artifact) if artifact is not None else None
    if domain is None:
        return "client_clock_missing"
    return build_server_clock(
        run_id=item.run_id,
        server_domain=item.clock_domain,
        client_domain=domain,
        recorded=artifact_alignments(records, item.run_id),
        offset_ns=offset,
        uncertainty_ns=uncertainty,
    )


def _process(
    item: LoadedEvidence,
    association: dict[str, Any],
    records: list[dict[str, Any]],
    artifact: dict[str, Any] | None,
    offset: int | None,
    uncertainty: int | None,
) -> dict[str, Any]:
    events = [
        event
        for event in item.events
        if event.event_type not in {"collector_degraded", "collector_recovered"}
    ]
    clock = _clock(item, records, artifact, offset, uncertainty)
    placements, unplaced = _placements(events, clock)
    windows = [
        record
        for record in records
        if record.get("event_type") == "infer.phase_window"
        and record.get("phase") == "measured"
    ]
    return {
        "engine": item.engine,
        "engine_version": item.manifest["engine"]["version"],
        "version_provenance": item.manifest["engine"]["version_provenance"],
        "process": item.manifest["process"],
        "capture_status": item.capture_status,
        "association": association,
        "memory": memory_summary(events),
        "clock": {
            "available": not isinstance(clock, str),
            "unavailable_reason": clock if isinstance(clock, str) else None,
            "provenance": None if isinstance(clock, str) else clock.evidence,
            "unaligned_samples": unplaced,
        },
        "cases": [
            _case_window(window, placements, item, association, clock)
            for window in windows
        ],
        "omlx_stats": stats_report(list(item.stats_records)),
        "warnings": list(item.warnings),
        "limitations": item.manifest["limitations"],
    }


def _placements(
    events: list[Any], clock: ServerClock | str
) -> tuple[list[tuple[Any, int, int]], int]:
    if isinstance(clock, str):
        return [], len(events)
    placed = []
    missing = 0
    for event in events:
        alignment = clock.align(event.timestamp_ns)
        if isinstance(alignment, str):
            missing += 1
        else:
            timestamp, _record = alignment
            placed.append((event, timestamp.value_ns, timestamp.uncertainty_ns))
    return placed, missing


def _case_window(
    window: dict[str, Any],
    placements: list[tuple[Any, int, int]],
    item: LoadedEvidence,
    association: dict[str, Any],
    clock: ServerClock | str,
) -> dict[str, Any]:
    result = {
        "case_id": window["case_id"],
        "association": association["state"],
        "scope": "process_during_case_window",
        "available": False,
        "unavailable_reason": None,
        "memory": None,
        "coverage": None,
    }
    if association["state"] in {"unjoined", "ambiguous"}:
        result["unavailable_reason"] = association["state"]
        return result
    if isinstance(clock, str):
        result["unavailable_reason"] = clock
        return result
    start, end = window.get("started_at_ns"), window.get("drained_at_ns")
    if type(start) is not int or type(end) is not int or end < start:
        result["unavailable_reason"] = "case_window_missing_or_invalid"
        return result
    inside, boundary = _window_samples(placements, start, end)
    result.update(
        available=bool(inside),
        unavailable_reason=None if inside else "no_definite_samples_in_window",
        memory=memory_summary([event for event, _ in inside]),
        coverage=_coverage(
            [timestamp for _, timestamp in inside], start, end, boundary, item
        ),
    )
    return result


def _window_samples(
    placements: list[tuple[Any, int, int]], start: int, end: int
) -> tuple[list[tuple[Any, int]], int]:
    inside = [
        (event, timestamp)
        for event, timestamp, uncertainty in placements
        if start <= timestamp - uncertainty and timestamp + uncertainty <= end
    ]
    boundary = sum(
        timestamp + uncertainty >= start and timestamp - uncertainty <= end
        for _, timestamp, uncertainty in placements
    ) - len(inside)
    return inside, boundary


def _coverage(
    timestamps: list[int], start: int, end: int, boundary: int, item: LoadedEvidence
) -> dict[str, Any]:
    ordered = sorted(timestamps)
    boundaries = [start, *ordered, end]
    interval = item.manifest["capture"]["interval_seconds"]
    expected = max(1, math.ceil((end - start) / (interval * 1e9)))
    return {
        "samples": len(ordered),
        "expected_at_nominal_cadence": expected,
        "uncertain_boundary_samples": boundary,
        "nominal_interval_seconds": interval,
        "window_seconds": (end - start) / 1e9,
        "max_observation_gap_seconds": max(
            (b - a for a, b in zip(boundaries, boundaries[1:])), default=0
        )
        / 1e9,
        "sample_density_ratio": min(1.0, len(ordered) / expected),
        "sampled_not_continuous": True,
        "capture_status": item.capture_status,
    }


def _metric_value(event: Any, path: str) -> int | float | None:
    value: Any = event
    for key in path.split("."):
        value = value.get(key) if isinstance(value, dict) else getattr(value, key, None)
    return (
        value
        if type(value) in (int, float) and math.isfinite(value) and value >= 0
        else None
    )


def memory_summary(events: list[Any]) -> dict[str, Any]:
    metrics = {}
    for name, (path, scope) in MEMORY_FIELDS.items():
        values = [
            value
            for event in events
            if (value := _metric_value(event, path)) is not None
        ]
        metrics[name] = {
            "unit": "bytes",
            "scope": scope,
            "valid_samples": len(values),
            "missing_samples": len(events) - len(values),
            "sampled_max": max(values) if values else None,
            "mean": sum(values) / len(values) if values else None,
            "counter_decreases": sum(b < a for a, b in zip(values, values[1:])),
        }
    return {
        "samples": len(events),
        "metrics": metrics,
        "device_total_bytes": None,
        "reserved_allocator_bytes": None,
        "device_free_bytes": None,
        "snapshot_consistency": "sequential_non_atomic",
        "peak_kind": "sampled_process_observations_not_exact_request_peaks",
        "memory_components_are_not_disjoint": True,
    }


def mlx_lines(report: dict[str, Any]) -> list[str]:
    mlx = report.get("telemetry", {}).get("mlx")
    if not isinstance(mlx, dict):
        return []
    lines = ["MLX evidence: sampled server processes; request ownership unavailable"]
    for process in mlx.get("processes", []):
        identity = process["process"]
        lines.append(
            f"- {process['engine']} PID {identity['pid']}: {process['association']['state']} ({process['association']['reason']})"
        )
        metrics = process["memory"]["metrics"]
        for name, metric in metrics.items():
            lines.append(
                f"  {name}: sampled max={metric['sampled_max']} bytes, n={metric['valid_samples']}, scope={metric['scope']}"
            )
        for case in process["cases"]:
            coverage = case["coverage"]
            detail = (
                f"{coverage['samples']} definite samples"
                if coverage
                else case["unavailable_reason"]
            )
            lines.append(f"  case {case['case_id']}: {detail}")
    stats = mlx.get("omlx_stats", {})
    if stats.get("samples"):
        lines.append("oMLX statistics: engine aggregate; other clients may contribute")
    return lines
