"""Analysis helpers for inference profiling artifacts."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, TypeGuard

from .telemetry import ServerIdentity, TelemetrySample, load_telemetry


def analyze_inference_events(
    path: str | Path,
    *,
    server_telemetry_paths: Iterable[str | Path] = (),
    direct_server: bool = False,
    clock_offset_ns: int | None = None,
    clock_uncertainty_ns: int | None = None,
) -> dict[str, Any]:
    """Analyze an inference profiling JSONL artifact."""
    records = _load_jsonl(path)
    requests, samples = _partition_inference_records(records)
    server_samples = [
        sample
        for telemetry_path in server_telemetry_paths
        for sample in load_telemetry(telemetry_path)
    ]
    join = _server_join(
        records,
        server_samples,
        direct_server=direct_server,
        clock_offset_ns=clock_offset_ns,
        clock_uncertainty_ns=clock_uncertainty_ns,
    )
    timeline = _server_timeline(server_samples, join)
    ok_requests = [record for record in requests if record.get("status") == "ok"]

    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in ok_requests:
        case_id = str(record.get("case_id", "unknown"))
        grouped.setdefault(case_id, []).append(record)

    cases = {}
    for case_id, case_requests in sorted(grouped.items()):
        cases[case_id] = _summarize_requests(
            case_requests,
            samples=_samples_for_request_window(samples, case_requests),
        )
        observations, coverage = _server_case_view(
            server_samples, case_requests, timeline
        )
        cases[case_id]["memory"]["server_observations"] = observations
        cases[case_id]["memory"]["server_coverage"] = coverage
    if timeline is not None:
        join["case_coverage"] = _coverage_counts(cases)
    failed = [record for record in requests if record.get("status") != "ok"]
    return {
        "summary": {
            "total_requests": len(requests),
            "successful_requests": len(ok_requests),
            "failed_requests": len(failed),
            "failure_rate": (len(failed) / len(requests)) if requests else 0.0,
            "case_count": len(cases),
        },
        "cases": cases,
        "telemetry": {
            "client_observation_scope": "client_local",
            "server_join": join,
            "server_targets": _server_targets(server_samples),
        },
    }


def format_analysis_text(report: dict[str, Any]) -> str:
    """Render an inference analysis report as text."""
    summary = report.get("summary", {})
    lines = [
        "Inference Profile Analysis",
        "-" * 28,
        f"Total requests: {summary.get('total_requests', 0)}",
        f"Successful requests: {summary.get('successful_requests', 0)}",
        f"Failed requests: {summary.get('failed_requests', 0)}",
        f"Failure rate: {float(summary.get('failure_rate', 0.0)):.2%}",
    ]
    cases = report.get("cases", {})
    join = report.get("telemetry", {}).get("server_join", {})
    lines.append("Memory observations: client-local")
    lines.extend(_server_status_lines(join))
    if isinstance(cases, dict) and cases:
        lines.append("")
        lines.append("Cases:")
        for case_id, case in cases.items():
            lines.extend(_case_lines(case_id, case))
    return "\n".join(lines)


def _server_status_lines(join: dict[str, Any]) -> list[str]:
    if join.get("status") == "joined":
        return _joined_status_lines(join)
    if join.get("status") not in {None, "not_configured"}:
        return [f"Server telemetry: unjoined ({join.get('reason')})"]
    return []


def _joined_status_lines(join: dict[str, Any]) -> list[str]:
    lines = [
        "Server telemetry: joined to case windows (declared direct route, "
        f"{join.get('clock_alignment_evidence')} clock evidence)"
    ]
    counts = join.get("case_coverage")
    if isinstance(counts, dict):
        lines.append(
            f"Server coverage: {counts.get('observed', 0)} observed, "
            f"{counts.get('partial', 0)} partial, {counts.get('empty', 0)} empty"
        )
    invalidation = join.get("invalidation")
    if isinstance(invalidation, dict):
        lines.append(
            f"Server identity ended ({invalidation.get('detail')}); "
            "case windows after the last confirmed poll are not joined"
        )
    return lines


def _case_lines(case_id: str, case: Any) -> list[str]:
    latency = case.get("latency_ms", {}) if isinstance(case, dict) else {}
    throughput = case.get("throughput", {}) if isinstance(case, dict) else {}
    lines = [
        f"- {case_id}: "
        f"p50 E2E={_fmt(latency.get('e2e_p50'))} ms, "
        f"p95 E2E={_fmt(latency.get('e2e_p95'))} ms, "
        f"p50 TTFT={_fmt(latency.get('ttft_p50'))} ms, "
        f"output={_fmt(throughput.get('output_tokens_per_second'))} tok/s, "
        f"requests={_fmt(throughput.get('requests_per_second'))} req/s"
    ]
    memory = case.get("memory", {}) if isinstance(case, dict) else {}
    lines.extend(_server_case_lines(memory))
    return lines


def _server_case_lines(memory: Any) -> list[str]:
    if not isinstance(memory, dict):
        return []
    coverage = memory.get("server_coverage") or {}
    if coverage.get("status") == "empty":
        return [f"  server telemetry: none ({coverage.get('reason')})"]
    lines = []
    if coverage.get("status") == "partial":
        lines.append(f"  server telemetry: partial ({coverage.get('reason')})")
    for metric, observation in (memory.get("server_observations") or {}).items():
        lines.append(_server_metric_line(metric, observation))
    return lines


def _server_metric_line(metric: str, observation: dict[str, Any]) -> str:
    label = f"  {metric} ({observation.get('observation_scope')})"
    value = observation.get("maximum_recorded_bytes")
    if value is None:
        missing = observation.get("missing_samples", 0)
        return f"{label}: no valid samples ({missing} missing)"
    return (
        f"{label}: max recorded {value} bytes, "
        f"{observation.get('valid_samples')} samples"
    )


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise TypeError(f"Line {line_number} is not a JSON object")
            records.append(payload)
    return records


def _summarize_requests(
    requests: list[dict[str, Any]],
    *,
    samples: list[dict[str, Any]],
) -> dict[str, Any]:
    e2e = _number_values(requests, "e2e_latency_ms")
    ttft = _number_values(requests, "ttft_ms")
    first_chunk = _number_values(requests, "first_chunk_latency_ms")
    output_tokens = sum(_int_value(record.get("output_tokens")) for record in requests)
    total_tokens = sum(_int_value(record.get("total_tokens")) for record in requests)
    request_window = _request_time_window(requests)
    duration_seconds = (
        max(request_window[1] - request_window[0], 0) / 1_000_000_000
        if request_window is not None
        else 0.0
    )
    request_count = len(requests)
    output_tps = output_tokens / duration_seconds if duration_seconds > 0 else 0.0
    total_tps = total_tokens / duration_seconds if duration_seconds > 0 else 0.0
    request_rate = request_count / duration_seconds if duration_seconds > 0 else 0.0
    peak_device_used = _peak_sample_value(samples, "device_used_bytes")
    peak_process_rss = _peak_sample_value(samples, "process_rss_bytes")
    return {
        "request_count": request_count,
        "latency_ms": {
            "e2e_p50": _percentile(e2e, 50),
            "e2e_p95": _percentile(e2e, 95),
            "e2e_p99": _percentile(e2e, 99),
            "ttft_p50": _percentile(ttft, 50),
            "ttft_p95": _percentile(ttft, 95),
            "ttft_p99": _percentile(ttft, 99),
            "first_chunk_p50": _percentile(first_chunk, 50),
            "first_chunk_p95": _percentile(first_chunk, 95),
        },
        "throughput": {
            "duration_seconds": duration_seconds,
            "requests_per_second": request_rate,
            "output_tokens_per_second": output_tps,
            "total_tokens_per_second": total_tps,
        },
        "tokens": {
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "output_token_sources": sorted(
                {
                    str(record.get("output_token_source", "unknown"))
                    for record in requests
                }
            ),
        },
        "memory": {
            "observation_scope": "client_local",
            "peak_device_used_bytes": peak_device_used,
            "peak_process_rss_bytes": peak_process_rss,
        },
    }


def _samples_for_request_window(
    samples: list[dict[str, Any]],
    requests: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    request_window = _request_time_window(requests)
    if request_window is None:
        return []

    start_ns, end_ns = request_window
    return [
        sample
        for sample in samples
        if _is_number(sample.get("timestamp_ns"))
        and start_ns <= _int_value(sample.get("timestamp_ns")) <= end_ns
    ]


def _request_time_window(requests: list[dict[str, Any]]) -> tuple[int, int] | None:
    bounds: list[tuple[int, int]] = []
    for record in requests:
        started_at = record.get("started_at_ns")
        ended_at = record.get("ended_at_ns")
        if not _is_number(started_at) or not _is_number(ended_at):
            continue
        bounds.append((_int_value(started_at), _int_value(ended_at)))
    if not bounds:
        return None
    return min(start for start, _end in bounds), max(end for _start, end in bounds)


def _peak_sample_value(samples: list[dict[str, Any]], field: str) -> int | None:
    return max(
        (
            _int_value(sample.get(field))
            for sample in samples
            if _is_number(sample.get(field))
        ),
        default=None,
    )


def _number_values(records: Iterable[dict[str, Any]], field: str) -> list[float]:
    values: list[float] = []
    for record in records:
        value = record.get(field)
        if _is_number(value):
            values.append(float(value))
    return values


def _percentile(values: list[float], percentile: int) -> float | None:
    if not values:
        return None
    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (percentile / 100.0) * (len(sorted_values) - 1)
    lower = int(rank)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = rank - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def _int_value(value: Any) -> int:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float):
        return int(value)
    return 0


def _is_number(value: Any) -> TypeGuard[int | float]:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _fmt(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.2f}"
    return "-"


def _partition_inference_records(
    records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    requests = [
        record
        for record in records
        if record.get("event_type") == "infer.request"
        and record.get("phase") == "measured"
    ]
    samples = [
        record
        for record in records
        if record.get("event_type") == "infer.system_sample"
    ]
    return requests, samples


def _server_join(
    records: list[dict[str, Any]],
    samples: list[TelemetrySample],
    *,
    direct_server: bool,
    clock_offset_ns: int | None,
    clock_uncertainty_ns: int | None,
) -> dict[str, Any]:
    """Return a case-window join only for one declared, matching server."""
    if not samples:
        return {"status": "not_configured"}
    artifact = _artifact_record(records)
    if artifact is None:
        return {"status": "unjoined", "reason": "missing_run_identity"}
    run_id_issue = _run_id_issue(artifact, samples)
    if run_id_issue is not None:
        return run_id_issue
    identities = {sample.identity for sample in samples}
    if len(identities) != 1:
        return {"status": "unjoined", "reason": "multiple_server_identities"}
    identity = next(iter(identities))
    route_issue = _route_issue(samples, direct_server)
    if route_issue:
        return {"status": "unjoined", "reason": route_issue}
    same_host = _same_host_clock(artifact, identity)
    alignment = _clock_alignment(same_host, clock_offset_ns, clock_uncertainty_ns)
    if "reason" in alignment:
        return {"status": "unjoined", "reason": alignment["reason"]}
    return {
        "status": "joined",
        "route_evidence": "operator_declared_direct",
        "identity": asdict(identity),
        **alignment,
        "attribution": "case_window_observation_only",
        "invalidation": _invalidation(samples),
    }


def _run_id_issue(
    artifact: dict[str, Any], samples: list[TelemetrySample]
) -> dict[str, Any] | None:
    """A different run ID leaves the server data out; the client report stays."""
    artifact_run_id = artifact["context"].get("run_id")
    telemetry_run_ids = sorted({sample.run_id for sample in samples})
    if telemetry_run_ids == [artifact_run_id]:
        return None
    return {
        "status": "unjoined",
        "reason": "run_id_mismatch",
        "artifact_run_id": artifact_run_id,
        "telemetry_run_ids": telemetry_run_ids,
    }


def _route_issue(samples: list[TelemetrySample], direct_server: bool) -> str | None:
    if not direct_server:
        return "route_not_declared"
    if not any(sample.state == "valid" for sample in samples):
        return "no_valid_server_samples"
    return None


def _invalidation(samples: list[TelemetrySample]) -> dict[str, Any] | None:
    """Locate where the observed process or GPU stopped being the original one.

    Samples before the first ``invalid`` sample describe the original identity,
    so an invalidation only affects case windows that could reach past the last
    poll that still confirmed it.
    """
    first = min(
        (sample for sample in samples if sample.state == "invalid"),
        key=lambda sample: sample.observed_at_ns,
        default=None,
    )
    if first is None:
        return None
    confirmed = [
        sample.observed_at_ns
        for sample in samples
        if sample.observed_at_ns < first.observed_at_ns
    ]
    return {
        "observed_at_ns": first.observed_at_ns,
        "detail": first.detail,
        "last_confirmed_at_ns": max(confirmed, default=None),
    }


def _artifact_record(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    artifacts = [r for r in records if r.get("event_type") == "infer.artifact"]
    if len(artifacts) != 1 or not isinstance(artifacts[0].get("context"), dict):
        return None
    return artifacts[0]


def _same_host_clock(artifact: dict[str, Any], identity: ServerIdentity) -> bool:
    metadata = artifact.get("metadata")
    if not isinstance(metadata, dict):
        return False
    boot_id = metadata.get("boot_id")
    return (
        artifact["context"].get("host") == identity.host
        and isinstance(boot_id, str)
        and bool(boot_id)
        and boot_id == identity.boot_id
    )


def _clock_alignment(
    same_host: bool,
    offset_ns: int | None,
    uncertainty_ns: int | None,
) -> dict[str, Any]:
    if offset_ns is None and not same_host:
        return {"reason": "clock_alignment_required"}
    if offset_ns is not None and uncertainty_ns is None:
        return {"reason": "clock_uncertainty_required"}
    # A shared clock needs no offset, but a supplied uncertainty is still honored.
    offset = 0 if offset_ns is None else offset_ns
    uncertainty = 0 if uncertainty_ns is None else uncertainty_ns
    _validate_clock_alignment(offset, uncertainty)
    return {
        "clock_offset_ns": offset,
        "clock_uncertainty_ns": uncertainty,
        "clock_alignment_evidence": (
            "same_host" if same_host and offset == 0 else "operator_supplied"
        ),
    }


def _validate_clock_alignment(offset_ns: int, uncertainty_ns: int) -> None:
    if isinstance(offset_ns, bool) or not isinstance(offset_ns, int):
        raise TypeError("clock_offset_ns must be an integer")
    if (
        isinstance(uncertainty_ns, bool)
        or not isinstance(uncertainty_ns, int)
        or uncertainty_ns < 0
    ):
        raise ValueError("clock_uncertainty_ns must be a non-negative integer")


def _server_targets(samples: list[TelemetrySample]) -> list[dict[str, Any]]:
    grouped: dict[ServerIdentity, list[TelemetrySample]] = {}
    for sample in samples:
        grouped.setdefault(sample.identity, []).append(sample)
    targets: list[dict[str, Any]] = []
    for identity in sorted(
        grouped,
        key=lambda item: (
            item.host,
            item.pid,
            item.process_start_ns,
            item.device_uuid or "",
        ),
    ):
        selected = grouped[identity]
        states = Counter(s.state for s in selected)
        targets.append(
            {
                "identity": asdict(identity),
                "metrics": sorted({s.metric for s in selected}),
                "sample_states": {
                    state: states[state]
                    for state in ("valid", "missing", "stale", "invalid")
                },
            }
        )
    return targets


@dataclass(frozen=True)
class _ServerTimeline:
    """A joined collector's polls in client time, and how long it can be trusted."""

    offset_ns: int
    uncertainty_ns: int
    slack_ns: int
    first_poll_ns: int
    last_poll_ns: int
    invalidated: bool
    trusted_until_ns: int | None


def _server_timeline(
    samples: list[TelemetrySample], join: dict[str, Any]
) -> _ServerTimeline | None:
    if join.get("status") != "joined":
        return None
    offset = join["clock_offset_ns"]
    uncertainty = join["clock_uncertainty_ns"]
    polls = [sample.observed_at_ns + offset for sample in samples]
    invalidation = join.get("invalidation")
    confirmed = invalidation.get("last_confirmed_at_ns") if invalidation else None
    return _ServerTimeline(
        offset_ns=offset,
        uncertainty_ns=uncertainty,
        slack_ns=max(sample.interval_ms for sample in samples) * 1_000_000,
        first_poll_ns=min(polls),
        last_poll_ns=max(polls),
        invalidated=invalidation is not None,
        # The identity was last confirmed at ``confirmed``; allow for clock error.
        trusted_until_ns=(
            None if confirmed is None else confirmed + offset - uncertainty
        ),
    )


def _server_case_view(
    samples: list[TelemetrySample],
    requests: list[dict[str, Any]],
    timeline: _ServerTimeline | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Summarize server samples in one case window and say how well it is covered.

    Only samples at least one clock uncertainty inside both window edges count,
    so a short window or a collector that was not running can leave a joined
    case without server values; the coverage record says which.
    """
    if timeline is None:
        return {}, {"status": "not_joined"}
    window = _request_time_window(requests)
    if window is None:
        return {}, {"status": "empty", "reason": "no_request_window"}
    low = window[0] + timeline.uncertainty_ns
    high = window[1] - timeline.uncertainty_ns
    in_window = [
        sample
        for sample in samples
        if low <= sample.observed_at_ns + timeline.offset_ns <= high
    ]
    coverage = _case_coverage(window[1], low, high, bool(in_window), timeline)
    if coverage["status"] == "empty":
        return {}, coverage
    return _metric_summaries(samples, in_window), coverage


def _case_coverage(
    end_ns: int,
    low: int,
    high: int,
    has_samples: bool,
    timeline: _ServerTimeline,
) -> dict[str, Any]:
    empty_reason = _empty_reason(end_ns, low, high, has_samples, timeline)
    if empty_reason is not None:
        return {
            "status": "empty",
            "reason": empty_reason,
            "counted_window_ns": [low, high] if high >= low else None,
        }
    partial_reason = _partial_reason(low, high, timeline)
    return {
        "status": "observed" if partial_reason is None else "partial",
        "reason": partial_reason,
        "counted_window_ns": [low, high],
    }


def _empty_reason(
    end_ns: int,
    low: int,
    high: int,
    has_samples: bool,
    timeline: _ServerTimeline,
) -> str | None:
    if high < low:
        return "window_shorter_than_uncertainty"
    if timeline.invalidated and (
        timeline.trusted_until_ns is None or end_ns > timeline.trusted_until_ns
    ):
        return "identity_invalidated"
    if not has_samples:
        return "no_collector_coverage"
    return None


def _partial_reason(low: int, high: int, timeline: _ServerTimeline) -> str | None:
    if timeline.first_poll_ns > low + timeline.slack_ns:
        return "collector_started_after_window_start"
    if timeline.last_poll_ns < high - timeline.slack_ns:
        return "collector_stopped_before_window_end"
    return None


def _coverage_counts(cases: dict[str, Any]) -> dict[str, int]:
    statuses = Counter(
        case["memory"]["server_coverage"]["status"] for case in cases.values()
    )
    return {status: statuses[status] for status in ("observed", "partial", "empty")}


def _metric_summaries(
    samples: list[TelemetrySample], in_window: list[TelemetrySample]
) -> dict[str, Any]:
    by_metric: dict[str, TelemetrySample] = {}
    for sample in samples:
        by_metric.setdefault(sample.metric, sample)
    window_by_metric: dict[str, list[TelemetrySample]] = {}
    for sample in in_window:
        window_by_metric.setdefault(sample.metric, []).append(sample)
    return {
        metric: _summarize_server_metric(first, window_by_metric.get(metric, []))
        for metric, first in sorted(by_metric.items())
    }


def _summarize_server_metric(
    first: TelemetrySample,
    window_samples: list[TelemetrySample],
) -> dict[str, Any]:
    """Describe one metric using only the samples inside the counted window."""
    states = Counter(sample.state for sample in window_samples)
    valid_values = [
        sample.value_bytes
        for sample in window_samples
        if sample.state == "valid" and sample.value_bytes is not None
    ]
    return {
        "observation_scope": first.scope,
        "counter_owner": first.counter_owner,
        "provenance": sorted({sample.provenance for sample in window_samples}),
        "sources": sorted({sample.source for sample in window_samples}),
        "maximum_recorded_bytes": max(valid_values, default=None),
        "valid_samples": len(valid_values),
        "missing_samples": states["missing"],
        "stale_samples": states["stale"],
        "invalid_samples": states["invalid"],
        "intervals_ms": sorted({sample.interval_ms for sample in window_samples}),
    }
